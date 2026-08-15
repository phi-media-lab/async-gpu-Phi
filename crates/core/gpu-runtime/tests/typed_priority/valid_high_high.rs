// should_compile: a real High task may await another spawned High typed task.

use gpu_runtime::executor::{ExecutorError, GpuExecutor};
use gpu_runtime::priority::{level::High, CompletionCell, PriorityClass};

static TARGET_RESULT: CompletionCell<u32> = CompletionCell::new();
static WAITER_RESULT: CompletionCell<()> = CompletionCell::new();

pub unsafe fn build_graph(executor: &GpuExecutor) -> Result<(), ExecutorError> {
    let target = executor.spawn_typed(PriorityClass::<High>::new(), &TARGET_RESULT, |_| async { 7 })?;
    let _waiter = executor.spawn_typed(
        PriorityClass::<High>::new(),
        &WAITER_RESULT,
        move |high| async move {
            let mut target = target;
            let _ = high.wait_for(&mut target).await;
        },
    )?;
    Ok(())
}
