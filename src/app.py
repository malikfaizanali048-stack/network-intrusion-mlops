from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse, HTMLResponse
from authlib.integrations.starlette_client import OAuth
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

load_dotenv()

app = FastAPI(title="Network Intrusion Detection API")
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET_KEY"))

oauth = OAuth()
oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")
USERS_FILE = Path("users.json")
BLOCKED_FILE = Path("blocked_users.json")

def load_json(path):
    if path.exists():
        return json.loads(path.read_text())
    return {}

def save_json(path, data):
    path.write_text(json.dumps(data, indent=2))

def record_login(email, name):
    users = load_json(USERS_FILE)
    now = datetime.utcnow().isoformat()
    if email not in users:
        users[email] = {"name": name, "first_login": now, "last_login": now, "login_count": 1}
    else:
        users[email]["last_login"] = now
        users[email]["login_count"] += 1
    save_json(USERS_FILE, users)

def is_blocked(email):
    blocked = load_json(BLOCKED_FILE)
    return email in blocked

def require_login(request: Request):
    user = request.session.get('user')
    if not user:
        return False
    if is_blocked(user.get('email')):
        return False
    return True

def require_admin(request: Request):
    user = request.session.get('user')
    return user and user.get('email') == ADMIN_EMAIL

Instrumentator().instrument(app).expose(app)

prediction_counter = Counter('predictions_total', 'Total predictions made', ['result'])
confidence_histogram = Histogram('prediction_confidence', 'Confidence scores of predictions')

MODEL_PATH = "models/random_forest_binary.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
model_loaded = model is not None


class TrafficFeatures(BaseModel):
    features: dict


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/model-info")
def model_info():
    if not model_loaded:
        return {"error": "Model not loaded"}
    return {
        "model_name": "network-intrusion-classifier",
        "version": "local-deployment",
        "source": "baked into Docker image for deployment; MLflow registry used for local governance workflow"
    }


@app.post("/predict")
def predict(data: TrafficFeatures):
    if model is None:
        return {"error": "Model not loaded"}

    df = pd.DataFrame([data.features])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]

    result = "ATTACK" if prediction == 1 else "BENIGN"
    confidence = float(max(probability))

    prediction_counter.labels(result=result).inc()
    confidence_histogram.observe(confidence)

    return {"prediction": result, "confidence": confidence}


@app.get("/explain")
def explain():
    try:
        with open("shap_feature_importance.json") as f:
            top_features = json.load(f)
        return {"top_features_driving_attack_predictions": top_features}
    except FileNotFoundError:
        return {"error": "SHAP explanation data not available"}


@app.get("/login")
async def login(request: Request):
    redirect_uri = request.url_for('auth_callback')
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth/callback")
async def auth_callback(request: Request):
    token = await oauth.google.authorize_access_token(request)
    user = token.get('userinfo')
    if not user:
        return HTMLResponse("<h2>Login failed.</h2>", status_code=400)
    if is_blocked(user.get('email')):
        return HTMLResponse("<h2>Access denied. This account has been blocked.</h2>", status_code=403)
    record_login(user.get('email'), user.get('name', ''))
    request.session['user'] = dict(user)
    return RedirectResponse(url='/dashboard')


@app.get("/logout")
async def logout(request: Request):
    request.session.pop('user', None)
    return RedirectResponse(url='/login')


@app.get("/dashboard")
async def dashboard(request: Request):
    if not require_login(request):
        return RedirectResponse(url='/login')
    with open("dashboard.html") as f:
        return HTMLResponse(f.read())


@app.get("/admin")
async def admin_panel(request: Request):
    if not require_admin(request):
        return HTMLResponse("<h2>Access denied. Admin only.</h2>", status_code=403)

    users = load_json(USERS_FILE)
    blocked = load_json(BLOCKED_FILE)

    rows = ""
    for email, info in users.items():
        status = "BLOCKED" if email in blocked else "active"
        action = "unblock" if email in blocked else "block"
        rows += f"""
        <tr>
          <td>{email}</td><td>{info['name']}</td><td>{info['login_count']}</td>
          <td>{info['last_login']}</td><td>{status}</td>
          <td><a href="/admin/{action}?email={email}">{action}</a></td>
        </tr>"""

    html = f"""
    <html><body style="font-family:sans-serif;padding:20px;">
    <h2>Admin — logged in users</h2>
    <table border="1" cellpadding="8" style="border-collapse:collapse;">
    <tr><th>Email</th><th>Name</th><th>Logins</th><th>Last login</th><th>Status</th><th>Action</th></tr>
    {rows}
    </table>
    </body></html>
    """
    return HTMLResponse(html)


@app.get("/admin/block")
async def block_user(request: Request, email: str):
    if not require_admin(request):
        return HTMLResponse("Access denied", status_code=403)
    blocked = load_json(BLOCKED_FILE)
    blocked[email] = True
    save_json(BLOCKED_FILE, blocked)
    return RedirectResponse(url='/admin')


@app.get("/admin/unblock")
async def unblock_user(request: Request, email: str):
    if not require_admin(request):
        return HTMLResponse("Access denied", status_code=403)
    blocked = load_json(BLOCKED_FILE)
    blocked.pop(email, None)
    save_json(BLOCKED_FILE, blocked)
    return RedirectResponse(url='/admin')