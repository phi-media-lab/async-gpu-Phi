// compile_fail: a real High task cannot await a spawned Normal typed task.

use gpu_runtime::executor::{ExecutorError, GpuExecutor};
use gpu_runtime::priority::{
    level::{High, Normal},
    CompletionCell, PriorityClass,
};

static NORMAL_RESULT: CompletionCell<u32> = CompletionCell::new();
static HIGH_RESULT: CompletionCell<()> = CompletionCell::new();

pub unsafe fn build_graph(executor: &GpuExecutor) -> Result<(), ExecutorError> {
    let normal = executor.spawn_typed(PriorityClass::<Normal>::new(), &NORMAL_RESULT, |_| async { 7 })?;
    let _high = executor.spawn_typed(PriorityClass::<High>::new(), &HIGH_RESULT, move |high| async move {
        let mut normal = normal;
        let _ = high.wait_for::<Normal, u32>(&mut normal).await;
    })?;
    Ok(())
}
