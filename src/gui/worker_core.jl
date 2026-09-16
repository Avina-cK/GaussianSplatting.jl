mutable struct FrameBuffer
    data::Array{Float32, 3} # (3, width, height)
    width::Int
    height::Int
end

FrameBuffer(width::Int, height::Int) =
    FrameBuffer(zeros(Float32, 3, width, height), width, height)

@enum WorkerActivity::Int32 begin
    ActivityIdle
    ActivityTraining
    ActivityRendering
    ActivityLoadingScene
    ActivitySaving
    ActivityExporting
    ActivityClosingScene
    ActivityPreparing
end

function activity_label(activity::WorkerActivity)
    activity ≡ ActivityTraining && return "Training step"
    activity ≡ ActivityRendering && return "Rendering"
    activity ≡ ActivityLoadingScene && return "Installing scene"
    activity ≡ ActivitySaving && return "Saving checkpoint"
    activity ≡ ActivityExporting && return "Exporting PLY"
    activity ≡ ActivityClosingScene && return "Closing scene"
    activity ≡ ActivityPreparing && return "Preparing training"
    return "Working"
end

"""
Scene-agnostic part of a GUI worker: a background task that owns all GPU work,
talking to the UI thread through a snapshot, a command channel and a swapped
double frame buffer.

The work itself is done by a handler `h` through the hooks:
`render_view!(h, snap, version)`, `worker_step!(h)`, `handle_command!(h, cmd)`,
`command_activity(h, tag)`, `render_enabled(h)`, `prepare_worker!(h)`.
"""
mutable struct WorkerCore{S}
    task::Maybe{Task}
    lock::ReentrantLock # Guards: snapshot, front/back swap, error_msg.
    wakeup::Base.Event
    commands::Channel{Any}

    snapshot::Maybe{S}
    snapshot_version::UInt64

    running::Threads.Atomic{Bool}

    front::FrameBuffer
    back::FrameBuffer
    has_new_frame::Bool
    front_version::UInt64
    back_version::UInt64

    activity::Threads.Atomic{Int32}
    activity_since::Threads.Atomic{Float64}

    error_msg::String
end

function WorkerCore{S}(; width::Int, height::Int) where S
    WorkerCore{S}(
        nothing, ReentrantLock(), Base.Event(true), Channel{Any}(32),
        nothing, UInt64(0),
        Threads.Atomic{Bool}(false),
        FrameBuffer(width, height), FrameBuffer(width, height), false,
        UInt64(0), UInt64(0),
        Threads.Atomic{Int32}(Int32(ActivityIdle)), Threads.Atomic{Float64}(0.0),
        "")
end

function render_view! end
function handle_command! end
worker_step!(_) = false
command_activity(_, ::Symbol) = ActivityRendering
render_enabled(_) = true
prepare_worker!(_) = nothing

function with_activity(f, core::WorkerCore, activity::WorkerActivity)
    # Timestamp first, so the UI never pairs a new activity with a stale time.
    core.activity_since[] = time()
    core.activity[] = Int32(activity)
    try
        return f()
    finally
        core.activity[] = Int32(ActivityIdle)
    end
end

function busy_status(core::WorkerCore)
    activity = WorkerActivity(core.activity[])
    activity ≡ ActivityIdle && return nothing
    return (; activity, elapsed=time() - core.activity_since[])
end

# The worker needs a `:default` thread not occupied by the GLFW loop.
function check_worker_threads()
    n_default = Threads.nthreads(:default)
    ok = Threads.threadpool() == :interactive ? n_default ≥ 1 : n_default ≥ 2
    ok || error(
        "GaussianSplatting GUI requires at least 2 Julia threads " *
        "to run training in the background. " *
        "Restart Julia with e.g. `julia -t 2,1` or `julia -t auto`.")
    return
end

function start_worker!(core::WorkerCore, h)
    prepare_worker!(h)
    core.running[] = true
    core.task = Threads.@spawn run_worker!(core, h)
    return
end

function stop_worker!(core::WorkerCore)
    core.running[] = false
    notify(core.wakeup)
    task = core.task
    core.task = nothing
    task ≡ nothing || wait(task)
    close(core.commands)
    return
end

wake!(core::WorkerCore) = notify(core.wakeup)

function submit!(core::WorkerCore, cmd::Tuple)
    put!(core.commands, cmd)
    notify(core.wakeup)
    return
end

function set_error!(core::WorkerCore, msg::String)
    lock(core.lock) do
        core.error_msg = msg
    end
    return
end

function take_error!(core::WorkerCore)::Maybe{String}
    lock(core.lock) do
        isempty(core.error_msg) && return nothing
        msg = core.error_msg
        core.error_msg = ""
        return msg
    end
end

# Returns the published version, see `fetch_frame`.
function publish!(core::WorkerCore{S}, snap::S)::UInt64 where S
    version = lock(core.lock) do
        core.snapshot = snap
        core.snapshot_version += 1
    end
    notify(core.wakeup)
    return version
end

latest_snapshot(core::WorkerCore) = lock(core.lock) do
    core.snapshot, core.snapshot_version
end

function back_buffer!(core::WorkerCore; width::Int, height::Int)
    back = core.back
    if back.width != width || back.height != height
        back.data = Array{Float32, 3}(undef, 3, width, height)
        back.width, back.height = width, height
    end
    return back.data
end

function swap_frames!(core::WorkerCore, version::UInt64)
    lock(core.lock) do
        core.front, core.back = core.back, core.front
        core.back_version, core.front_version = core.front_version, version
        core.has_new_frame = true
    end
    return
end

# UI thread only.
function upload_frame!(core::WorkerCore, surface; width::Int, height::Int)
    lock(core.lock) do
        core.has_new_frame || return
        frame = core.front
        # Stale size: the worker renders a matching one from the next snapshot.
        (frame.width == width && frame.height == height) || return
        NGL.set_data!(surface, frame.data)
        core.has_new_frame = false
        return
    end
    return
end

# Copy of the latest frame once it was rendered from `version` or newer.
function fetch_frame(core::WorkerCore, version::UInt64; width::Int, height::Int)
    lock(core.lock) do
        core.front_version ≥ version || return nothing
        frame = core.front
        (frame.width == width && frame.height == height) || return nothing
        return copy(frame.data)
    end
end

function run_worker!(core::WorkerCore, h)
    last_version = UInt64(0)
    last_view_time = 0.0
    while core.running[]
        try
            drain_commands!(core, h) && (last_version = UInt64(0))

            did_step = worker_step!(h)
            snap, version = latest_snapshot(core)

            did_render = false
            # While stepping, refresh the view at ≤ 10 FPS.
            if render_enabled(h) && snap ≢ nothing &&
                (version != last_version || (did_step && time() - last_view_time > 0.1))
                with_activity(core, ActivityRendering) do
                    render_view!(h, snap, version)
                end
                last_version = version
                last_view_time = time()
                did_render = true
            end

            did_step || did_render || wait(core.wakeup)
        catch err
            set_error!(core, "Render worker error. See logs for details.")
            @error "Render worker error:" exception=(err, catch_backtrace())
            wait(core.wakeup)
        end
    end
    return
end

function drain_commands!(core::WorkerCore, h)
    rerender = false
    while isready(core.commands)
        cmd = take!(core.commands)::Tuple
        try
            rerender |= with_activity(core, command_activity(h, cmd[1]::Symbol)) do
                handle_command!(h, cmd)
            end
        catch err
            set_error!(core, "`$(first(cmd))` failed. See logs for details.")
            @error "Worker command failed:" cmd=first(cmd) exception=(err, catch_backtrace())
        end
    end
    return rerender
end
