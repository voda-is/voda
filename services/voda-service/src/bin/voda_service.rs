use anyhow::Result;
use axum::Router;
use tokio::sync::mpsc;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use voda_runtime::define_function_types;
use voda_runtime::ExecutableFunctionCall;
use voda_runtime::FunctionExecutor;
use voda_runtime_roleplay::RoleplayRuntimeClient;
use voda_service_api::{
    character_routes, misc_routes, setup_tracing, system_config_routes, user_routes, 
    runtime_routes, voice_routes, memory_routes
};

use voda_service::voda_routes;
use voda_runtime_evm::GitcoinFunctionCall;
define_function_types! {
    Gitcoin(GitcoinFunctionCall, "gitcoin_allocate_grant")
}

#[tokio::main]
async fn main() -> Result<()> {
    setup_tracing();

    let cors = CorsLayer::very_permissive();
    let trace = TraceLayer::new_for_http();

    let (executor, execution_queue) = mpsc::channel(100);
    let client = RoleplayRuntimeClient::new(executor).await;

    tokio::spawn(async move {
        let mut function_executor = FunctionExecutor::<RuntimeFunctionType>::new(
            execution_queue
        );
        function_executor.run().await;
    });

    let app = Router::new()

        // generic routes
        .merge(character_routes())
        .merge(user_routes())
        .merge(misc_routes())
        .merge(memory_routes())
        .merge(system_config_routes())
        .merge(runtime_routes())
        .merge(voice_routes())

        // voda routes
        .merge(voda_routes())

        .layer(cors)
        .layer(trace)
        .with_state(client);

    let port: u16 = std::env::var("PORT")
        .unwrap_or("3033".into())
        .parse()
        .expect("failed to convert to number");

    let listener = tokio::net::TcpListener::bind(format!(":::{port}"))
        .await
        .unwrap();

    tracing::info!("LISTENING ON {port}");
    axum::serve(listener, app.into_make_service()).await.unwrap();
    Ok(())
}
