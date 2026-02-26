"""
make_ml_dataset.py  –  ML Dataset Generator (v2)

ceo_data.json (merge_history.py 実行後) から機械学習用データセットを生成します。

生成ファイル:
  data/ml_dataset.csv       - 全CEO × 株価 × 写真パス (1行 = 1CEO在任期間)
  data/face_embeddings.npz  - 顔埋め込みベクトル (FaceNet/ArcFace 512次元)
  data/face_analysis.csv    - PCA + 超過リターン回帰結果
  data/stats.json           - データセット統計

Usage:
  python scripts/make_ml_dataset.py [--embed] [--analyze]
  --embed   : DeepFace で顔埋め込みを生成 (要 deepface インストール)
  --analyze : PCA + Ridge 回帰で顔寄与度分析
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR   = Path(r"C:\Users\hp\Documents\CEO_Photos_Project")
DATA_DIR      = PROJECT_DIR / "data"
CEO_DATA_FILE = DATA_DIR / "ceo_data.json"
ML_CSV        = DATA_DIR / "ml_dataset.csv"
FACE_NPZ      = DATA_DIR / "face_embeddings.npz"
ANALYSIS_CSV  = DATA_DIR / "face_analysis.csv"
STATS_JSON    = DATA_DIR / "stats.json"


def safe_name(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s\u3000]', "_", str(s)).strip("_")


def photos_dir(ticker: str) -> Path:
    try:
        if int(ticker) < 5500:
            return PROJECT_DIR / "photos_1"
    except ValueError:
        pass
    return PROJECT_DIR / "photos_2"


def _excess_return(row: dict) -> float | None:
    """超過リターン (年換算) = 株価リターン - TOPIX リターン"""
    try:
        open_s  = float(row.get("open_at_appointment") or 0)
        close_s = float(row.get("close_at_resignation") or 0)
        open_t  = float(row.get("topix_open_at_appointment") or 0)
        close_t = float(row.get("topix_close_at_resignation") or 0)
        if not (open_s > 0 and close_s > 0 and open_t > 0 and close_t > 0):
            return None
        appt = row.get("appointment_date", "")
        res  = row.get("resignation_date", "")
        if appt and res:
            try:
                d_appt = datetime.fromisoformat(str(appt)[:10])
                d_res  = datetime.fromisoformat(str(res)[:10])
                years  = max((d_res - d_appt).days / 365.25, 0.1)
            except Exception:
                years = 3.0  # フォールバック
        else:
            years = 3.0
        stock_ret = (close_s / open_s) ** (1 / years) - 1
        topix_ret = (close_t / open_t) ** (1 / years) - 1
        return round(stock_ret - topix_ret, 6)
    except Exception:
        return None


def _find_photo_paths(ticker: str, comp_dir: str, role: str) -> list[str]:
    """photos_1/ or photos_2/ から写真パスを列挙 (GitHub相対パス)。"""
    base = photos_dir(ticker) / comp_dir / role
    if not base.exists():
        return []
    return [
        str(p.relative_to(PROJECT_DIR)).replace("\\", "/")
        for p in sorted(base.glob("photo_*.jpg"))
    ]


def build_dataset(companies: list[dict]) -> pd.DataFrame:
    rows = []

    for company in companies:
        ticker   = company.get("ticker", "")
        cname    = company.get("company_name", "")
        url      = company.get("url", "")
        sn       = safe_name(cname)
        comp_dir = f"{ticker}_{sn}"

        # ── 現役社長 ──
        ceo = company.get("current_ceo") or {}
        if ceo.get("name"):
            photos = _find_photo_paths(ticker, comp_dir, "current")
            if not photos:
                # ceo_data.json の photo_path フィールドを参照
                saved = ceo.get("photos_saved") or (["photo_01.jpg"] if ceo.get("photo_saved") else [])
                photos = [
                    f"photos_1/{comp_dir}/current/{p}" if int(ticker or 0) < 5500
                    else f"photos_2/{comp_dir}/current/{p}"
                    for p in saved
                ]
            rows.append({
                "ticker":                    ticker,
                "company_name":              cname,
                "url":                       url,
                "ceo_name":                  ceo.get("name", ""),
                "ceo_role":                  "current",
                "appointment_date":          ceo.get("appointment_info") or ceo.get("appointment_date", ""),
                "resignation_date":          None,
                "open_at_appointment":       ceo.get("open_at_appointment", ""),
                "close_at_resignation":      None,
                "topix_open_at_appointment": ceo.get("topix_open_at_appointment", ""),
                "topix_close_at_resignation": None,
                "excess_return_annual":      None,  # 在任中のため計算不可
                "tenure_years":              None,
                "photo_paths":               ",".join(photos),
                "photo_count":               len(photos),
                "source":                    "current",
            })

        # ── 歴代社長 ──
        for i, prev in enumerate(company.get("previous_ceos") or [], 1):
            name = prev.get("name", "")
            if not name:
                continue
            pn = safe_name(name)
            hist_role = f"history/{i:02d}_{pn}"
            photos = _find_photo_paths(ticker, comp_dir, hist_role)
            if not photos and prev.get("photo_path"):
                photos = [str(prev["photo_path"]).replace("\\", "/")]

            # 在任年数
            tenure = None
            try:
                d1 = datetime.fromisoformat(str(prev.get("appointment_date", ""))[:10])
                d2 = datetime.fromisoformat(str(prev.get("resignation_date", ""))[:10])
                tenure = round((d2 - d1).days / 365.25, 2)
            except Exception:
                pass

            row = {
                "ticker":                    ticker,
                "company_name":              cname,
                "url":                       url,
                "ceo_name":                  name,
                "ceo_role":                  f"prev_{i}",
                "appointment_date":          prev.get("appointment_date", ""),
                "resignation_date":          prev.get("resignation_date", ""),
                "open_at_appointment":       prev.get("open_at_appointment", ""),
                "close_at_resignation":      prev.get("close_at_resignation", ""),
                "topix_open_at_appointment": prev.get("topix_open_at_appointment", ""),
                "topix_close_at_resignation": prev.get("topix_close_at_resignation", ""),
                "excess_return_annual":      None,
                "tenure_years":              tenure,
                "photo_paths":               ",".join(photos),
                "photo_count":               len(photos),
                "source":                    "history",
            }
            row["excess_return_annual"] = _excess_return(row)
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ── 顔埋め込み生成 ──────────────────────────────────────────────────────────
def extract_face_embeddings(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    DeepFace (FaceNet/ArcFace) で写真から512次元顔ベクトルを抽出。
    photos_1/ or photos_2/ のローカルファイルが必要。

    Returns:
        embeddings: shape (N, 512)
        indices:    DataFrameの行インデックス文字列
    """
    try:
        from deepface import DeepFace
    except ImportError:
        print("DeepFace not installed. Run: pip install deepface")
        return np.array([]), []

    MODEL = "ArcFace"  # ArcFace が最高精度 (FaceNet512 も可)
    embeddings = []
    indices    = []

    for idx, row in df[df["photo_count"] > 0].iterrows():
        first_photo = row["photo_paths"].split(",")[0]
        photo_path  = PROJECT_DIR / first_photo
        if not photo_path.exists():
            continue
        try:
            result = DeepFace.represent(
                img_path     = str(photo_path),
                model_name   = MODEL,
                enforce_detection = False,
                detector_backend  = "retinaface",
            )
            if result:
                vec = np.array(result[0]["embedding"])
                embeddings.append(vec)
                indices.append(str(idx))
        except Exception as e:
            print(f"  Skip {first_photo}: {e}")

    if embeddings:
        return np.stack(embeddings), indices
    return np.array([]), []


# ── PCA + 回帰分析 ──────────────────────────────────────────────────────────
def analyze_face_contribution(df: pd.DataFrame, embeddings: np.ndarray, indices: list[str]):
    """
    PCA で顔ベクトルを次元削減 → Ridge 回帰で超過リターンへの寄与度を測定。

    目的変数  y : 在任中の超過リターン (年換算) = 株価CAGRリターン - TOPIX CAGRリターン
    説明変数  X_base  : 業種 + 時代 + 在任年数 (コントロール変数)
             X_face  : PCA 主成分 (10〜20次元)
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model  import Ridge, RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.pipeline import Pipeline

    # 分析対象: 歴代社長 × 超過リターンあり × 埋め込みあり
    df_hist = df[
        (df["source"] == "history") &
        (df["excess_return_annual"].notna()) &
        (df.index.astype(str).isin(indices))
    ].copy()

    if len(df_hist) < 30:
        print(f"分析サンプル不足 ({len(df_hist)}件). 30件以上必要。")
        return None

    print(f"\n分析対象: {len(df_hist)} CEO × 就任-退任期間")

    # 顔埋め込みを対応付け
    idx_to_emb = {str(i): emb for i, emb in zip(indices, embeddings)}
    face_vecs  = np.stack([idx_to_emb[str(i)] for i in df_hist.index])

    # 目的変数
    y = df_hist["excess_return_annual"].values.astype(float)

    # ── ベースライン特徴量 ──
    # 時代 (就任年を5年ごとにbin)
    df_hist["era"] = df_hist["appointment_date"].str[:4].apply(
        lambda y: f"era_{int(y)//5*5}" if y and y.isdigit() else "era_unknown"
    )
    # 業種はtickerから大まかに推定 (理想はSIC/業種コード)
    df_hist["sector_code"] = (df_hist["ticker"].str[:2].str.zfill(2))

    X_base_df = pd.get_dummies(
        df_hist[["era", "sector_code"]], drop_first=True
    )
    if "tenure_years" in df_hist.columns:
        X_base_df["tenure_years"] = df_hist["tenure_years"].fillna(df_hist["tenure_years"].median())

    X_base = X_base_df.values.astype(float)

    # ── 顔PCA ──
    N_COMPONENTS = min(20, face_vecs.shape[0] // 5, face_vecs.shape[1])
    scaler = StandardScaler()
    face_scaled = scaler.fit_transform(face_vecs)

    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    X_face = pca.fit_transform(face_scaled)
    expl   = pca.explained_variance_ratio_
    print(f"PCA {N_COMPONENTS}成分: 顔分散の {expl.sum():.1%} を説明")

    X_full = np.hstack([X_base, X_face])

    # ── Cross-validation (5-fold) ──
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    cv     = KFold(n_splits=5, shuffle=True, random_state=42)

    def cv_r2(X):
        model = RidgeCV(alphas=alphas, cv=5)
        return cross_val_score(model, X, y, cv=cv, scoring="r2")

    r2_base = cv_r2(X_base)
    r2_full = cv_r2(X_full)

    delta = r2_full.mean() - r2_base.mean()
    print(f"\nBaseline R²: {r2_base.mean():.4f} ± {r2_base.std():.4f}")
    print(f"Full R²:     {r2_full.mean():.4f} ± {r2_full.std():.4f}")
    print(f"顔の寄与 ΔR²: {delta:.4f}")
    print(f"  (正 = 顔情報が株価パフォーマンスを説明, 負 = ノイズ)")

    # ── 係数解釈: 主要PC成分の回帰係数 ──
    model_full = Ridge(alpha=1.0)
    model_full.fit(X_full, y)
    face_coefs = model_full.coef_[X_base.shape[1]:]  # 顔成分の係数

    results = {
        "n_samples":        int(len(df_hist)),
        "n_face_pca":       int(N_COMPONENTS),
        "face_variance_explained": float(expl.sum()),
        "baseline_r2_mean": float(r2_base.mean()),
        "baseline_r2_std":  float(r2_base.std()),
        "full_r2_mean":     float(r2_full.mean()),
        "full_r2_std":      float(r2_full.std()),
        "face_delta_r2":    float(delta),
        "top_face_pc_coefs": face_coefs[:5].tolist(),
    }

    # CSV 保存
    result_rows = []
    for i, coef in enumerate(face_coefs):
        result_rows.append({
            "pc": i + 1,
            "coefficient": round(coef, 6),
            "explained_var": round(float(expl[i]), 4),
        })
    pd.DataFrame(result_rows).to_csv(ANALYSIS_CSV, index=False, encoding="utf-8-sig")
    print(f"\n分析結果: {ANALYSIS_CSV}")
    return results


# ── メイン ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed",   action="store_true", help="顔埋め込み生成")
    parser.add_argument("--analyze", action="store_true", help="PCA + 回帰分析")
    args = parser.parse_args()

    if not CEO_DATA_FILE.exists():
        print("ceo_data.json が見つかりません。先に merge_history.py を実行してください。")
        sys.exit(1)

    companies = json.loads(CEO_DATA_FILE.read_text(encoding="utf-8"))
    print(f"企業数: {len(companies)}")

    # ── データセット構築 ──
    df = build_dataset(companies)
    print(f"総CEO行数: {len(df)} (現役: {(df['source']=='current').sum()}, 歴代: {(df['source']=='history').sum()})")
    print(f"写真あり: {(df['photo_count']>0).sum()}")
    print(f"超過リターン計算可能: {df['excess_return_annual'].notna().sum()}")

    df.to_csv(ML_CSV, index=False, encoding="utf-8-sig")
    print(f"\nML dataset: {ML_CSV}")

    # ── 統計 ──
    hist = df[df["source"] == "history"]
    stats = {
        "generated_at":             datetime.now().isoformat(),
        "total_companies":          len(companies),
        "total_ceo_rows":           int(len(df)),
        "current_ceo_rows":         int((df["source"] == "current").sum()),
        "history_ceo_rows":         int((df["source"] == "history").sum()),
        "with_photo":               int((df["photo_count"] > 0).sum()),
        "with_excess_return":       int(df["excess_return_annual"].notna().sum()),
        "excess_return_mean":       round(float(hist["excess_return_annual"].dropna().mean()), 4)
                                    if hist["excess_return_annual"].notna().any() else None,
        "excess_return_std":        round(float(hist["excess_return_annual"].dropna().std()), 4)
                                    if hist["excess_return_annual"].notna().any() else None,
        "coverage_photo_pct":       f"{(df['photo_count']>0).mean()*100:.1f}%",
    }
    STATS_JSON.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== Stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # ── 顔埋め込み生成 ──
    embeddings, indices = np.array([]), []
    if args.embed:
        print("\n顔埋め込み生成中 (ArcFace)...")
        embeddings, indices = extract_face_embeddings(df)
        if len(embeddings):
            np.savez(FACE_NPZ, embeddings=embeddings, indices=np.array(indices))
            print(f"顔埋め込み保存: {FACE_NPZ} ({len(embeddings)}件)")
    elif FACE_NPZ.exists():
        loaded     = np.load(FACE_NPZ, allow_pickle=True)
        embeddings = loaded["embeddings"]
        indices    = list(loaded["indices"])
        print(f"顔埋め込み読み込み: {len(embeddings)}件")

    # ── 回帰分析 ──
    if args.analyze and len(embeddings) > 0:
        results = analyze_face_contribution(df, embeddings, indices)
        if results:
            print(f"\n顔寄与度 ΔR²: {results['face_delta_r2']:.4f}")


if __name__ == "__main__":
    main()
