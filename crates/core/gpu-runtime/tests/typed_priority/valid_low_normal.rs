// should_compile: a real Low task may await a spawned Normal typed task.

use gpu_runtime::executor::{ExecutorError, GpuExecutor};
use gpu_runtime::priority::{
    level::{Low, Normal},
    CompletionCell, PriorityClass,
};

static TARGET_RESULT: CompletionCell<u32> = CompletionCell::new();
static WAITER_RESULT: CompletionCell<()> = CompletionCell::new();

pub unsafe fn build_graph(executor: &GpuExecutor) -> Result<(), ExecutorError> {
    let target =
        executor.spawn_typed(PriorityClass::<Normal>::new(), &TARGET_RESULT, |_| async {
            7
        })?;
    let _waiter = executor.spawn_typed(
        PriorityClass::<Low>::new(),
        &WAITER_RESULT,
        move |low| async move {
            let mut target = target;
            let _ = low.wait_for(&mut target).await;
        },
    )?;
    Ok(())
}
