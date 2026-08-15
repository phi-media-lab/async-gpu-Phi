// should_compile: a real Low task may await a spawned High typed task.

use gpu_runtime::executor::{ExecutorError, GpuExecutor};
use gpu_runtime::priority::{
    level::{High, Low},
    CompletionCell, PriorityClass,
};

static HIGH_RESULT: CompletionCell<u32> = CompletionCell::new();
static LOW_RESULT: CompletionCell<()> = CompletionCell::new();

pub unsafe fn build_graph(executor: &GpuExecutor) -> Result<(), ExecutorError> {
    let high = executor.spawn_typed(PriorityClass::<High>::new(), &HIGH_RESULT, |_| async { 7 })?;
    let _low = executor.spawn_typed(PriorityClass::<Low>::new(), &LOW_RESULT, move |low| async move {
        let mut high = high;
        let _ = low.wait_for(&mut high).await;
    })?;
    Ok(())
}
