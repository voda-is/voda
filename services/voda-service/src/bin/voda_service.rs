use std::sync::Arc;

use anyhow::Result;
use axum::Router;
use tower_http::{cors::CorsLayer, timeout::TimeoutLayer, trace::TraceLayer};
use reqwest;

use voda_service_api::{
    graphql_route, misc_routes, runtime_routes, setup_tracing, voice_routes, user_routes, GlobalState
};

use voda_runtime_mem0::{init_pgvector_pool, Mem0Engine};
use voda_database::init_db_pool;
use voda_runtime::{SystemConfig, User, UserBadge, UserReferral, UserUrl, UserUsage, Memory};
use voda_runtime_character_creation::{CharacterCreationMessage, CharacterCreationRuntimeClient};
use voda_runtime_roleplay::{AuditLog, Character, RoleplayMessage, RoleplayRuntimeClient, RoleplaySession};

init_db_pool!(
    User, UserUsage, UserUrl, UserReferral, UserBadge, SystemConfig,
    Character, RoleplaySession, RoleplayMessage, AuditLog,
    CharacterCreationMessage
);

init_pgvector_pool!();

#[tokio::main]
async fn main() -> Result<()> {
    setup_tracing();

    let cors = CorsLayer::very_permissive();
    let trace = TraceLayer::new_for_http();

    let db_pool = Arc::new(connect(false, false).await.clone());
    let pgvector_db = Arc::new(connect_pgvector(false, false).await.clone());

    let (roleplay_client, mut mem0_messages_rx) = RoleplayRuntimeClient::new(db_pool.clone(), pgvector_db.clone()).await?;
    let character_creation_client = CharacterCreationRuntimeClient::new(db_pool.clone(), "character_creation_v0".to_string()).await?;

    let global_state = GlobalState {
        roleplay_client: roleplay_client,
        character_creation_client: character_creation_client,
        http_client: reqwest::Client::new(),
    };

    tokio::spawn(async move {
        let mem0 = Mem0Engine::new(db_pool.clone(), pgvector_db.clone()).await
            .expect("[Mem0Engine::new] Failed to create mem0 engine");
        while let Some(mem0_messages) = mem0_messages_rx.recv().await {
            let adding_result = mem0.add_messages(&mem0_messages).await;
            if let Err(e) = adding_result {
                tracing::warn!("[Mem0Engine::add_messages] Failed to add messages: {:?}", e);
            }
        }
    });

    let app = Router::new()
        .merge(misc_routes())
        .merge(runtime_routes())
        .merge(voice_routes())
        .merge(graphql_route())
        .merge(user_routes())
        .layer(TimeoutLayer::new(std::time::Duration::from_secs(3600)))
        .layer(cors)
        .layer(trace)
        .with_state(global_state);

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
