# Immutable view request published by the UI thread.
# The camera is a deepcopy: the worker treats it as read-only.
struct ViewSnapshot
    camera::Camera
    sh_degree::Int # -1 means use value from the model.
    mode::Int # 0 - color, 1 - depth.
    background::SVector{3, Float32} # Color the splats are composited over.
end

"""
GaussianSplatting side of the GUI worker (see `WorkerCore`), with `GSGUI` as the handler.
"""
mutable struct RenderWorker
    core::WorkerCore{ViewSnapshot}

    # UI → worker control flags.
    train::Threads.Atomic{Bool}
    densify::Threads.Atomic{Bool}
    render::Threads.Atomic{Bool}
    # Step to stop training at; `0` means no limit.
    max_steps::Threads.Atomic{Int}
    # Periodic checkpointing (see `maybe_autosave!`). `autosave_path` is a
    # `String`, so unlike the flags around it, it is guarded by `core.lock`
    # (see `autosave_path` & `set_autosave_path!`).
    autosave::Threads.Atomic{Bool}
    autosave_every::Threads.Atomic{Int}
    autosave_path::String

    # worker → UI stats.
    loss::Threads.Atomic{Float32}
    loss_ema::Threads.Atomic{Float32} # See `smoothed_total`.
    # Wall time spent inside `step!` & the steps it covers, both since this
    # scene was installed. Time *in* the steps, not since training started: the
    # worker also renders views & sits idle, and a checkpoint carries no record
    # of what its earlier run cost.
    train_time::Threads.Atomic{Float64}
    train_steps::Threads.Atomic{Int}
    step::Threads.Atomic{Int}
    n_gaussians::Threads.Atomic{Int}
    memory::Threads.Atomic{Int} # Device bytes; see `refresh_memory!`.

    # worker → UI, rare (under `core.lock`).
    pick_result::Maybe{SVector{3, Float32}}
    # Per-term loss curves for the plot: a copy of the trainer's `LossHistory`,
    # republished whenever it takes a new sample. The UI thread must not read
    # the trainer's own vectors, which the worker keeps appending to.
    loss_history::Maybe{NamedTuple}

    # Worker-local: the snapshot of the last render, so commands that
    # inspect `rast.image` (orbit picking) know the matching camera.
    last_snapshot::Maybe{ViewSnapshot}
end

function RenderWorker(; width::Int, height::Int)
    RenderWorker(
        WorkerCore{ViewSnapshot}(; width, height),
        # train, densify, render, max_steps
        Threads.Atomic{Bool}(false), Threads.Atomic{Bool}(true), Threads.Atomic{Bool}(true),
        Threads.Atomic{Int}(Int(DEFAULT_MAX_STEPS)),
        # autosave, autosave_every, autosave_path
        Threads.Atomic{Bool}(false),
        Threads.Atomic{Int}(Int(DEFAULT_AUTOSAVE_EVERY)), "",
        # loss, loss_ema, train_time, train_steps, step, n_gaussians, memory
        Threads.Atomic{Float32}(0f0), Threads.Atomic{Float32}(0f0),
        Threads.Atomic{Float64}(0.0), Threads.Atomic{Int}(0),
        Threads.Atomic{Int}(0), Threads.Atomic{Int}(0), Threads.Atomic{Int}(0),
        # pick_result, loss_history, last_snapshot
        nothing, nothing, nothing)
end

"""
Republish the trainer's loss curves for the UI thread, if they grew since the
last publish. Copying is what makes them safe to read off-thread, so it only
happens on the steps that actually sampled (see `LossHistory`).
"""
function publish_loss_history!(w::RenderWorker, trainer)
    history = trainer.losses.history
    published = lock(w.core.lock) do
        w.loss_history ≡ nothing ? -1 : w.loss_history.version
    end
    published == history.version && return

    published_history = snapshot(history)
    lock(w.core.lock) do
        w.loss_history = published_history
    end
    return
end

# Latest published loss curves, or `nothing` before the first sample.
loss_history(w::RenderWorker) = lock(w.core.lock) do
    w.loss_history
end

function take_pick_result!(w::RenderWorker)::Maybe{SVector{3, Float32}}
    lock(w.core.lock) do
        result = w.pick_result
        w.pick_result = nothing
        return result
    end
end

"""
Publish the device memory the scene holds (see `memory_usage`).

Computed on the worker: it walks arrays that densification replaces, so the
UI thread must not do it itself. Called wherever the scene changes size —
after a train step & on install/close.
"""
function refresh_memory!(gui, w::RenderWorker)
    w.memory[] =
        memory_usage(gui.gaussians) +
        memory_usage(gui.trainer) +
        memory_usage(gui.rasterizer) +
        memory_usage(gui.sky_rasterizer)
    return
end

# Publish the current camera & render settings for the worker to render.
# Returns the version of the published snapshot, so that callers which
# need this exact frame back (video capture) can wait for it.
function publish_view!(gui)::UInt64
    w = gui.worker
    snap = ViewSnapshot(
        deepcopy(gui.camera),
        Int(gui.ui_state.sh_degree[]),
        Int(gui.ui_state.selected_mode[]),
        SVector{3, Float32}(gui.ui_state.background_color))
    return publish!(w.core, snap)
end

"""
Whether the trainer may take another step.
A non-positive `max_steps` means unlimited number of training steps.
"""
training_steps_left(w::RenderWorker, trainer) =
    w.max_steps[] ≤ 0 || trainer.step < w.max_steps[]

autosave_path(w::RenderWorker) = lock(() -> w.autosave_path, w.core.lock)

function set_autosave_path!(w::RenderWorker, path::String)
    lock(w.core.lock) do
        w.autosave_path = path
    end
    return
end

"""
`base` with the step number appended, so a run leaves an ordered series behind
instead of overwriting itself: `scene.safetensors` at step 7000 becomes
`scene_007000.safetensors`.
"""
function autosave_filename(base::String, step::Int)
    stem = endswith(base, ".safetensors") ? base[1:end - length(".safetensors")] : base
    return "$(stem)_$(lpad(step, 6, '0')).safetensors"
end

"""
Write a checkpoint on the autosave schedule: every `autosave_every` steps, and
on the run's final step, so an unattended run always leaves one behind.

Called right after a step rather than next to the loop's stop check, which runs
on every idle iteration & would save the same state over and over. A failed
write switches autosave off: the next one is `autosave_every` steps of training
away and would fail the same way.
"""
function maybe_autosave!(w::RenderWorker, trainer)
    w.autosave[] || return
    every = w.autosave_every[]
    is_final = !training_steps_left(w, trainer)
    (is_final || (every > 0 && trainer.step % every == 0)) || return

    base = autosave_path(w)
    isempty(base) && return
    filename = autosave_filename(base, trainer.step)
    try
        with_activity(w.core, ActivitySaving) do
            save_state(trainer, filename)
        end
        @info "Autosaved at step $(trainer.step): `$filename`."
    catch err
        w.autosave[] = false
        set_error!(w.core, "Autosave failed, autosave turned off. See logs for details.")
        @error "Autosave to `$filename` failed, turning autosave off." exception=(err, catch_backtrace())
    end
    return
end

# Push the UI toggles to the worker flags (after `reset_ui!`).
function sync_worker_flags!(gui)
    w = gui.worker
    ui_state = gui.ui_state
    w.train[] = ui_state.train[]
    w.densify[] = ui_state.densify[]
    w.render[] = ui_state.render[]
    w.max_steps[] = Int(ui_state.max_steps[])
    w.autosave[] = ui_state.autosave[]
    w.autosave_every[] = Int(ui_state.autosave_every[])
    set_autosave_path!(w, ui_state.autosave_path)
    wake!(w.core)
    return
end

function train_step!(gui, w::RenderWorker)
    trainer = gui.trainer
    trainer ≡ nothing && return false
    w.train[] && !training_steps_left(w, trainer) && (w.train[] = false)
    w.train[] || return false

    trainer.densify = w.densify[]
    try
        step_started = time()
        loss = with_activity(w.core, ActivityTraining) do
            step!(trainer)
        end
        # Only the worker writes these, so a plain read-modify-write
        # is enough; the UI thread only reads them.
        w.train_time[] += time() - step_started
        w.train_steps[] += 1
        w.loss[] = loss
        # Each step scores a different view, so the raw loss jitters
        # on view difficulty: the average is what shows a trend.
        w.loss_ema[] = smoothed_total(trainer.losses)
        publish_loss_history!(w, trainer)
        w.step[] = trainer.step
        w.n_gaussians[] = length(gui.gaussians)
        refresh_memory!(gui, w)
        # After the stats: an autosave is slow enough that the UI
        # should show this step's numbers while it runs.
        maybe_autosave!(w, trainer)

        # The single loss scalar hides which term is moving, which
        # is what matters once several regularizers compete for the
        # same scales & opacities (see `LossBreakdown`).
        if trainer.step % LOSS_REPORT_INTERVAL == 0
            println("step $(trainer.step) | ↓ loss=$(round(loss; digits=5)) | " *
                format_breakdown(trainer.losses.current))
            println("            ema($LOSS_EMA_HORIZON) | " *
                format_breakdown(smoothed(trainer.losses)))
        end
    catch err
        # E.g. non-finite loss:
        # stop training instead of killing the worker, the scene stays viewable.
        w.train[] = false
        set_error!(w.core, "Training step failed, stopping training. See logs for details.")
        @error "Training step failed, stopping training." exception=(err, catch_backtrace())
    end
    return true
end

function scene_command_activity(tag::Symbol)
    (tag ≡ :install_scene || tag ≡ :install_model) && return ActivityLoadingScene
    tag ≡ :save_state && return ActivitySaving
    tag ≡ :export_ply && return ActivityExporting
    tag ≡ :close_scene && return ActivityClosingScene
    return ActivityRendering # `:pick_orbit` reads the rendered depth.
end

"""
Handle commands send by GUI.
Return `true` if any of the commands requires re-rendering of the scene.
"""
function handle_scene_command!(gui, w::RenderWorker, cmd::Tuple)
    tag = cmd[1]::Symbol
    if tag ≡ :install_scene
        loaded = cmd[2]
        gui.gaussians = loaded.gaussians
        gui.rasterizer = loaded.gui_rasterizer
        gui.sky_rasterizer = loaded.gui_sky_rasterizer
        gui.trainer = loaded.trainer
        w.loss[] = 0f0
        w.loss_ema[] = 0f0
        # The new scene's curves start empty: the old ones are another run.
        lock(w.core.lock) do
            w.loss_history = nothing
        end
        w.train_time[] = 0.0
        w.train_steps[] = 0
        w.step[] = loaded.trainer.step
        w.n_gaussians[] = length(loaded.gaussians)
        refresh_memory!(gui, w)
        return true
    elseif tag ≡ :install_model
        gaussians = cmd[2]::GaussianModel
        gui.gaussians = gaussians
        # Viewer-only mode: a loaded PLY already has the dome merged into it.
        gui.trainer = nothing
        gui.sky_rasterizer = nothing
        w.loss[] = 0f0
        w.loss_ema[] = 0f0
        # The new scene's curves start empty: the old ones are another run.
        lock(w.core.lock) do
            w.loss_history = nothing
        end
        w.train_time[] = 0.0
        w.train_steps[] = 0
        w.step[] = 0
        w.n_gaussians[] = length(gaussians)
        refresh_memory!(gui, w)
        return true
    elseif tag ≡ :close_scene
        return handle_close_scene!(gui, w)
    elseif tag ≡ :save_state
        gui.trainer ≡ nothing || save_state(gui.trainer, cmd[2]::String)
        return false
    elseif tag ≡ :export_ply
        gs = gui.gaussians
        if gs ≢ nothing
            # Fold the dome in: it is part of the scene's appearance, and a PLY
            # has nowhere else to put it.
            sky = gui.trainer ≡ nothing ? nothing : gui.trainer.sky
            export_ply(sky ≡ nothing ? gs : merge_sky(gs, sky), cmd[2]::String)
        end
        return false
    elseif tag ≡ :pick_orbit
        handle_pick!(gui, w, cmd[2]::Int, cmd[3]::Int)
        return true
    end
    error("Unknown worker command: `$tag`.")
end

"""
Drop the current scene & release the device memory it holds.
Returns `true` to signal `handle_scene_command!` that the scene needs to be redrawn.
"""
function handle_close_scene!(gui, w::RenderWorker)
    w.train[] = false

    trainer, gaussians = gui.trainer, gui.gaussians
    sky_rast = gui.sky_rasterizer
    gui.trainer, gui.gaussians, gui.sky_rasterizer = nothing, nothing, nothing

    trainer ≡ nothing || KA.unsafe_free!(trainer)
    gaussians ≡ nothing || KA.unsafe_free!(gaussians)
    sky_rast ≡ nothing || KA.unsafe_free!(sky_rast)
    release_scene_buffers!(gui.rasterizer)

    w.loss[] = 0f0
    w.loss_ema[] = 0f0
    lock(w.core.lock) do
        w.loss_history = nothing
    end
    w.train_time[] = 0.0
    w.train_steps[] = 0
    w.step[] = 0
    w.n_gaussians[] = 0
    # `rast.image` no longer holds the depth an orbit pick would unproject.
    w.last_snapshot = nothing
    refresh_memory!(gui, w)

    GC.gc(false)
    GC.gc(true)
    return true
end

# Render the view described by `snap` (published as `version`) into the
# back buffer & swap.
function render_scene_view!(gui, w::RenderWorker, snap::ViewSnapshot, version::UInt64)
    camera = snap.camera
    (; width, height) = resolution(camera)

    rast = gui.rasterizer
    if size(rast.image)[2:3] != (width, height)
        kab = get_backend(rast)
        # TODO free the old one before creating new one.
        gui.rasterizer = rast = GaussianRasterizer(kab, camera; mode=:rgbd)
    end

    trainer = gui.trainer
    sky = trainer ≡ nothing ? nothing : trainer.sky
    sky_rast = gui.sky_rasterizer
    if sky ≢ nothing && (sky_rast ≡ nothing || size(sky_rast.image)[2:3] != (width, height))
        gui.sky_rasterizer = sky_rast =
            sky_view_rasterizer(get_backend(rast), sky, camera)
    end

    back = back_buffer!(w.core; width, height)

    gs = gui.gaussians
    if gs ≡ nothing || length(gs) == 0
        # Empty scene (no dataset loaded yet): display background color.
        for c in 1:3
            @view(back[c, :, :]) .= snap.background[c]
        end
    else
        sh_degree = snap.sh_degree == -1 ? gs.sh_degree : snap.sh_degree
        rast(
            gs.points, gs.opacities, gs.scales,
            gs.rotations, gs.features_dc, gs.features_rest;
            camera, sh_degree, background=snap.background)
        # Only touches the color rows, so the depth view & orbit picking below
        # still read pure scene geometry.
        sky ≡ nothing || composite_sky!(rast, sky, camera; sky_rast)
        tex = snap.mode == 1 ? gl_depth(rast) : gl_texture(rast)
        # `tex` is the rasterizer-owned host buffer, overwritten by the next render,
        # so copy it out before publishing.
        copyto!(back, tex)
    end

    w.last_snapshot = snap
    swap_frames!(w.core, version)
    # The first render allocates the scene-sized scratch buffers, and a
    # resize rebuilds the rasterizer: both move the number above.
    refresh_memory!(gui, w)
    return
end

"""
Pick a new orbiting target by unprojecting the rendered depth under
the `(px, py)` pixel (1-based, top-left origin).
Depth is averaged over a small window around the pixel to be robust
to outliers at fuzzy silhouettes; background pixels (nothing rendered
there, so the blended depth is ≈ 0) are excluded.
Uses the camera of the last-rendered snapshot, matching the contents
of `rast.image`; the result is published to `pick_result`.
"""
function handle_pick!(gui, w::RenderWorker, px::Integer, py::Integer)
    snap = w.last_snapshot
    snap ≡ nothing && return

    gs = gui.gaussians
    (gs ≡ nothing || length(gs) == 0) && return

    rast = gui.rasterizer
    rast.mode == :rgbd || return

    camera = snap.camera
    (; width, height) = resolution(camera)
    (1 ≤ px ≤ width && 1 ≤ py ≤ height) || return

    δ = 4 # Window half-size.
    depths = Array(@view(rast.image[4,
        max(1, px - δ):min(width, px + δ),
        max(1, py - δ):min(height, py + δ)]))
    valid = depths .> 1f-2 # Near plane: rejects background pixels.
    any(valid) || return
    z = mean(depths[valid])

    # Unproject through the camera intrinsics
    # (COLMAP frame: x right, y down, z forward).
    fx, fy = camera.intrinsics.focal
    cx = camera.intrinsics.principal[1] * width
    cy = camera.intrinsics.principal[2] * height
    p_cam = SVector{3, Float32}(
        (px - 0.5f0 - cx) * z / fx,
        (py - 0.5f0 - cy) * z / fy,
        z)
    R = SMatrix{3, 3, Float32}(@view(camera.c2w[1:3, 1:3]))
    target = R * p_cam .+ view_pos(camera)

    lock(w.core.lock) do
        w.pick_result = target
    end
    return
end
