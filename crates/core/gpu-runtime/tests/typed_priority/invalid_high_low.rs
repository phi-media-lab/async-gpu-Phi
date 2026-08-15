// compile_fail: a real High task cannot await a spawned Low typed task.

use gpu_runtime::executor::{ExecutorError, GpuExecutor};
use gpu_runtime::priority::{
    level::{High, Low},
    CompletionCell, PriorityClass,
};

static LOW_RESULT: CompletionCell<u32> = CompletionCell::new();
static HIGH_RESULT: CompletionCell<()> = CompletionCell::new();

pub unsafe fn build_graph(executor: &GpuExecutor) -> Result<(), ExecutorError> {
    let low = executor.spawn_typed(PriorityClass::<Low>::new(), &LOW_RESULT, |_| async { 7 })?;
    let _high = executor.spawn_typed(PriorityClass::<High>::new(), &HIGH_RESULT, move |high| async move {
        let mut low = low;
        let _ = high.wait_for::<Low, u32>(&mut low).await;
    })?;
    Ok(())
}
