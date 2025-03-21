use anyhow::anyhow;
use axum::{
    extract::{Extension, Path, State}, 
    http::StatusCode, middleware, 
    routing::{get, post}, Json, Router
};
use voda_database::{doc, MongoDbObject};
use voda_common::{blake3_hash, get_current_timestamp, CryptoHash, EnvVars};

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::env::ApiServerEnv;
use crate::middleware::{authenticate, admin_only};
use crate::response::{AppError, AppSuccess};

use voda_common::encrypt;
use voda_runtime::{ExecutableFunctionCall, RuntimeClient, User, UserProfile, UserProvider};

pub const AMOUNT_PER_CLAIM: u64 = 100;

pub fn user_routes<S: RuntimeClient<F>, F: ExecutableFunctionCall>() -> Router<S> {
    Router::new()
        // TODO: deprecate this
        .route("/token",post(generate_token)
            .route_layer(middleware::from_fn(admin_only))
        )
        .route("/user", post(save_user::<S, F>)
            .route_layer(middleware::from_fn(authenticate))
        )

        .route("/user/{user_id}", get(get_user::<S, F>))
        .route("/users", post(get_users::<S, F>))
        .route("/user/claim_points/{user_id}", post(claim_free_points::<S, F>))
}

#[derive(Debug, Serialize, Deserialize)]
struct UserPayload {
    user_id: String,
    network_name: Option<String>,
    username: String,
    first_name: String,
    last_name: Option<String>,
    profile_photo: Option<String>,
}

async fn save_user<S: RuntimeClient<F>, F: ExecutableFunctionCall>(
    State(state): State<S>,
    Extension(user_id): Extension<CryptoHash>,
    Json(payload): Json<UserPayload>,
) -> Result<AppSuccess, AppError> {
    let provider_str = payload.user_id.split(":").next().ok_or(AppError::new(StatusCode::BAD_REQUEST, anyhow!("Invalid provider")))?;
    let provider = match provider_str {
        "telegram" => UserProvider::Telegram,
        "google" => UserProvider::Google,
        "x" => UserProvider::X,
        "crypto_wallet" => UserProvider::CryptoWallet,
        _ => return Err(AppError::new(StatusCode::BAD_REQUEST, anyhow!("Invalid provider"))),
    };

    let payload_user_id = blake3_hash(payload.user_id.as_bytes());
    if payload_user_id != user_id {
        return Err(AppError::new(StatusCode::UNAUTHORIZED, anyhow!("User ID mismatch")));
    }
    let old_user = User::select_one_by_index(&state.get_db(), &payload_user_id).await?;
    match old_user {
        Some(mut old_user) => {
            old_user.user_id = payload.user_id;
            old_user.provider = provider;
            old_user.network_name = payload.network_name;
            old_user.profile.first_name = payload.first_name;
            old_user.profile.last_name = payload.last_name;
            old_user.profile.username = payload.username;
            old_user.profile.avatar = payload.profile_photo;
            old_user.last_active = get_current_timestamp();
            old_user.populate_id();
            old_user.update(&state.get_db()).await?;
        }
        None => {
            let profile = UserProfile {
                id: payload_user_id,
                user_personality: vec![],
                username: payload.username,
                first_name: payload.first_name,
                last_name: payload.last_name,
                avatar: payload.profile_photo,
                bio: None,
                email: None,
                phone: None,
            };

            let mut new_user = User::new(profile, payload.user_id.clone());
            new_user.user_id = payload.user_id;
            new_user.provider = provider;
            new_user.network_name = payload.network_name;
            new_user.populate_id();
            new_user.save(&state.get_db()).await?;
        }
    }

    Ok(AppSuccess::new(
        StatusCode::CREATED,
        "User created successfully",
        json!(())
    ))
}

async fn get_user<S: RuntimeClient<F>, F: ExecutableFunctionCall>(
    State(state): State<S>,
    Path(user_id): Path<CryptoHash>,
) -> Result<AppSuccess, AppError> {
    let user = User::select_one_by_index(&state.get_db(), &user_id).await?
        .ok_or(AppError::new(StatusCode::NOT_FOUND, anyhow!("User not found")))?;

    Ok(AppSuccess::new(
        StatusCode::OK,
        "User fetched successfully",
        json!(user)
    ))
}

#[derive(Debug, Serialize, Deserialize)]
struct GetUsersRequest {
    user_ids: Vec<String>,
}
async fn get_users<S: RuntimeClient<F>, F: ExecutableFunctionCall>(
    State(state): State<S>,
    Json(request): Json<GetUsersRequest>,
) -> Result<AppSuccess, AppError> {
    let user_ids = request.user_ids;
    if user_ids.is_empty() {
        return Err(AppError::new(StatusCode::BAD_REQUEST, anyhow!("User ID is required")));
    }
    
    let users = User::select_many_simple(&state.get_db(), doc! { "_id": { "$in": user_ids } }).await?;
    Ok(AppSuccess::new(StatusCode::OK, "Users fetched successfully", json!(users)))
}

// TODO: deprecate this
async fn generate_token(
    Json(user_id): Json<String>,
) -> Result<AppSuccess, AppError> {
    let env = ApiServerEnv::load();
    let encrypted = encrypt(&user_id, &env.secret_salt)?;
    Ok(AppSuccess::new(
        StatusCode::OK, 
        "Token generated successfully", 
        json!(encrypted)
    ))
}

async fn claim_free_points<S: RuntimeClient<F>, F: ExecutableFunctionCall>(
    State(state): State<S>,
    Path(user_id): Path<CryptoHash>,
) -> Result<AppSuccess, AppError> {
    User::claim_free_balance(&state.get_db(), &user_id, AMOUNT_PER_CLAIM).await?;
    Ok(AppSuccess::new(StatusCode::OK, "Points claimed successfully", json!(())))
}
