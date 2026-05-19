import ast
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import webbrowser
import os
import json
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns

TRAIN_PATH = "data/train.csv"
TEST_PATH  = "data/test.csv"

SUCCESS_LABELS = {
    "Very Positive",
    "Overwhelmingly Positive",
    "Mostly Positive",
    "Positive",
}
FAILURE_LABELS = {
    "Negative",
    "Mostly Negative",
    "Mixed",
    "Very Negative",
    "Overwhelmingly Negative",
}

TOP_TAGS_COUNT = 50


def load_and_filter(path):
    df = pd.read_csv(path)
    valid = SUCCESS_LABELS | FAILURE_LABELS
    df = df[df["review_summary"].isin(valid)].copy()
    df["success"] = df["review_summary"].isin(SUCCESS_LABELS).astype(int)
    return df


def parse_list_col(series):
    def safe_parse(val):
        try:
            return ast.literal_eval(val)
        except Exception:
            return []
    return series.apply(safe_parse)


def get_top_tags(train_df, n):
    tag_counts = {}
    for tag_list in parse_list_col(train_df["tags"]):
        for tag in tag_list:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    sorted_tags = sorted(tag_counts, key=tag_counts.get, reverse=True)
    return sorted_tags[:n]


def encode_tags(df, top_tags):
    tag_lists = parse_list_col(df["tags"])
    tag_set_list = [set(lst) for lst in tag_lists]
    encoded = {}
    for tag in top_tags:
        col_name = f"tag_{tag.lower().replace(' ', '_')}"
        encoded[col_name] = [1 if tag in tag_set else 0 for tag_set in tag_set_list]
    return pd.DataFrame(encoded, index=df.index)


def build_numeric_features(df):
    df = df.copy()
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year.fillna(0).astype(int)

    df["tag_count"]   = parse_list_col(df["tags"]).apply(len)
    df["genre_count"] = parse_list_col(df["genres"]).apply(len)
    df["lang_count"]  = parse_list_col(df["languages"]).apply(len)
    df["is_free"]     = (df["price"] == 0).astype(int)

    price_cap = 5000
    df["price_capped"] = df["price"].clip(0, price_cap)

    cols = ["price_capped", "is_free", "tag_count",
            "genre_count", "lang_count", "release_year"]
    return df[cols].fillna(0)


def print_results(model_name, y_test, y_pred):
    total = len(y_test)
    correct = accuracy_score(y_test, y_pred, normalize=False)
    acc = correct / total

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    success_total = tp + fn
    failure_total = tn + fp
    success_correct_pct = tp / success_total * 100 if success_total > 0 else 0
    failure_correct_pct = tn / failure_total * 100 if failure_total > 0 else 0

    print(f"загвар: {model_name}")
    print(f"нийт {total} тоглоомоос {int(correct)} зөв таарсан ({acc*100:.1f}%)")
    print()
    print(f"  амжилттай тоглоом:  {success_total}-аас {tp} зөв  ({success_correct_pct:.1f}%)")
    print(f"  амжилтгүй тоглоом:  {failure_total}-аас {tn} зөв  ({failure_correct_pct:.1f}%)")
    print()
    print(f"  алдаа 1 - амжилттайг амжилтгүй гэж таасан: {fn}")
    print(f"  алдаа 2 - амжилтгүйг амжилттай гэж таасан: {fp}")


def run_multinomial_nb(x_train, y_train, x_test, y_test):
    model = MultinomialNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print_results("Naive Bayes - MultinomialNB (tag feature)", y_test, y_pred)
    return model


def run_gaussian_nb(x_train, y_train, x_test, y_test):
    model = GaussianNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print_results("Naive Bayes - GaussianNB (numeric feature)", y_test, y_pred)
    return model


# ─────────────────────────────────────────────
#  VISUAL DASHBOARD - HTML REPORT
# ─────────────────────────────────────────────

def generate_visual_report(results_data):
    """
    results_data dict:
      - game_info:   dict of user inputs
      - pred:        0 or 1
      - prob:        [fail_prob, success_prob]
      - mnb_pred, mnb_prob
      - gnb_pred, gnb_prob
      - factors:     list of (label, type) tuples
      - train_stats: dict with success_rate etc.
    """
    gi = results_data["game_info"]
    pred = results_data["pred"]
    prob = results_data["prob"]
    mnb_pred = results_data["mnb_pred"]
    mnb_prob = results_data["mnb_prob"]
    gnb_pred = results_data["gnb_pred"]
    gnb_prob = results_data["gnb_prob"]
    factors = results_data["factors"]

    success_pct = round(prob[1] * 100, 1)
    fail_pct = round(prob[0] * 100, 1)
    result_label = "АМЖИЛТТАЙ" if pred == 1 else "АМЖИЛТГҮЙ"
    result_color = "#1D9E75" if pred == 1 else "#D85A30"
    result_bg    = "#E1F5EE" if pred == 1 else "#FAECE7"

    def model_badge(p, prob_arr):
        lbl = "SUCCESS" if p == 1 else "FAILURE"
        bg  = "#E1F5EE" if p == 1 else "#FAECE7"
        col = "#0F6E56" if p == 1 else "#993C1D"
        conf = round(max(prob_arr) * 100, 1)
        return f'<span style="background:{bg};color:{col};padding:3px 10px;border-radius:6px;font-size:13px;font-weight:500;">{lbl} {conf}%</span>'

    dt_badge  = model_badge(pred, prob)
    mnb_badge = model_badge(mnb_pred, mnb_prob)
    gnb_badge = model_badge(gnb_pred, gnb_prob)

    votes = [pred, mnb_pred, gnb_pred]
    majority = 1 if sum(votes) >= 2 else 0
    ensemble_label = "АМЖИЛТТАЙ" if majority == 1 else "АМЖИЛТГҮЙ"
    ensemble_color = "#1D9E75" if majority == 1 else "#D85A30"

    factors_html = ""
    for label, ftype in factors[:6]:
        icon = "✓" if ftype == "positive" else "⚠"
        fc   = "#0F6E56" if ftype == "positive" else "#854F0B"
        fbg  = "#E1F5EE" if ftype == "positive" else "#FAEEDA"
        factors_html += f'''
        <div style="display:flex;align-items:center;gap:10px;padding:8px 12px;
                    background:{fbg};border-radius:8px;margin-bottom:6px;">
          <span style="color:{fc};font-size:16px;font-weight:bold;">{icon}</span>
          <span style="color:{fc};font-size:14px;">{label}</span>
        </div>'''

    tags_str    = ", ".join(gi.get("tags", [])) or "—"
    genres_str  = ", ".join(gi.get("genres", [])) or "—"
    langs_str   = ", ".join(gi.get("languages", [])) or "—"
    price_str   = f"${float(gi.get('price', 0)):.2f}"
    date_str    = gi.get("release_date", "—")

    html = f"""<!DOCTYPE html>
<html lang="mn">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Game Success Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #f5f4ef;
    color: #2C2C2A;
    min-height: 100vh;
    padding: 32px 16px;
  }}
  .container {{ max-width: 860px; margin: 0 auto; }}
  h1 {{
    font-size: 22px; font-weight: 600; letter-spacing: -0.5px;
    margin-bottom: 4px; color: #2C2C2A;
  }}
  .subtitle {{ font-size: 14px; color: #888780; margin-bottom: 28px; }}
  .card {{
    background: #fff;
    border: 0.5px solid #D3D1C7;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 20px;
  }}
  .card-title {{
    font-size: 13px; font-weight: 500; color: #888780;
    text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 16px;
  }}
  .result-hero {{
    display: flex; align-items: center; gap: 20px;
    padding: 28px; border-radius: 14px;
    background: {result_bg}; border: 0.5px solid #D3D1C7;
    margin-bottom: 20px;
  }}
  .result-icon {{
    font-size: 52px; line-height: 1;
  }}
  .result-label {{
    font-size: 28px; font-weight: 700; color: {result_color};
    letter-spacing: -1px;
  }}
  .result-sub {{
    font-size: 14px; color: #5F5E5A; margin-top: 4px;
  }}
  .metrics-grid {{
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;
    margin-bottom: 20px;
  }}
  .metric-card {{
    background: #fff; border: 0.5px solid #D3D1C7;
    border-radius: 10px; padding: 16px 18px;
    text-align: center;
  }}
  .metric-label {{
    font-size: 12px; color: #888780; margin-bottom: 6px;
  }}
  .metric-value {{
    font-size: 26px; font-weight: 600; color: #2C2C2A;
  }}
  .info-grid {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
  }}
  .info-row {{
    display: flex; flex-direction: column; gap: 2px;
    padding: 10px 14px; background: #f5f4ef; border-radius: 8px;
  }}
  .info-key {{ font-size: 11px; color: #888780; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }}
  .info-val {{ font-size: 14px; color: #2C2C2A; }}
  .model-row {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 0; border-bottom: 0.5px solid #f0efe8;
  }}
  .model-row:last-child {{ border-bottom: none; }}
  .model-name {{ font-size: 14px; color: #2C2C2A; }}
  .ensemble-box {{
    display: flex; align-items: center; gap: 12px;
    padding: 16px 20px; border-radius: 10px;
    background: #f5f4ef; margin-top: 16px;
  }}
  .ensemble-label {{ font-size: 13px; color: #5F5E5A; }}
  .ensemble-value {{ font-size: 16px; font-weight: 600; color: {ensemble_color}; }}
  @media (max-width: 520px) {{
    .metrics-grid {{ grid-template-columns: 1fr 1fr; }}
    .info-grid {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div class="container">

  <h1>🎮 Game Success Report</h1>
  <p class="subtitle">Machine Learning prediction dashboard — Decision Tree + Naive Bayes ensemble</p>

  <!-- RESULT HERO -->
  <div class="result-hero">
    <div class="result-icon">{"✅" if pred == 1 else "❌"}</div>
    <div>
      <div class="result-label">{result_label}</div>
      <div class="result-sub">Decision Tree загварын таамаглал · {success_pct}% амжилтын магадлал</div>
    </div>
  </div>

  <!-- PROBABILITY METRICS -->
  <div class="metrics-grid">
    <div class="metric-card">
      <div class="metric-label">Амжилтын магадлал</div>
      <div class="metric-value" style="color:#1D9E75;">{success_pct}%</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Амжилтгүйн магадлал</div>
      <div class="metric-value" style="color:#D85A30;">{fail_pct}%</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Ensemble санал</div>
      <div class="metric-value" style="color:{ensemble_color};font-size:18px;">{ensemble_label}</div>
    </div>
  </div>

  <!-- PROBABILITY CHART -->
  <div class="card">
    <div class="card-title">Магадлалын харьцуулалт</div>
    <div style="display:flex;gap:12px;margin-bottom:12px;font-size:12px;color:#888780;">
      <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#1D9E75;margin-right:4px;"></span>Амжилттай</span>
      <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#D85A30;margin-right:4px;"></span>Амжилтгүй</span>
    </div>
    <div style="position:relative;width:100%;height:100px;">
      <canvas id="probChart" role="img"
        aria-label="Probability bar chart showing success {success_pct}% and failure {fail_pct}%">
        Амжилт: {success_pct}%, Амжилтгүй: {fail_pct}%
      </canvas>
    </div>
  </div>

  <!-- MODEL COMPARISON -->
  <div class="card">
    <div class="card-title">Загваруудын харьцуулалт</div>
    <div class="model-row">
      <span class="model-name">Decision Tree</span>
      {dt_badge}
    </div>
    <div class="model-row">
      <span class="model-name">Multinomial Naive Bayes</span>
      {mnb_badge}
    </div>
    <div class="model-row">
      <span class="model-name">Gaussian Naive Bayes</span>
      {gnb_badge}
    </div>
    <div class="ensemble-box">
      <span class="ensemble-label">🗳 Majority vote (ensemble):</span>
      <span class="ensemble-value">{ensemble_label}</span>
    </div>
  </div>

  <!-- KEY FACTORS -->
  <div class="card">
    <div class="card-title">Гол хүчин зүйлүүд</div>
    {"".join([f'<div style="display:flex;align-items:center;gap:10px;padding:8px 12px;background:{"#E1F5EE" if t=="positive" else "#FAEEDA"};border-radius:8px;margin-bottom:6px;"><span style="color:{"#0F6E56" if t=="positive" else "#854F0B"};font-size:16px;font-weight:bold;">{"✓" if t=="positive" else "⚠"}</span><span style="color:{"#0F6E56" if t=="positive" else "#854F0B"};font-size:14px;">{l}</span></div>' for l,t in factors[:6]] or ["<p style='color:#888780;font-size:14px;'>Тодорхой хүчин зүйл илрэгдсэнгүй</p>"])}
  </div>

  <!-- GAME INFO -->
  <div class="card">
    <div class="card-title">Тоглоомын мэдээлэл</div>
    <div class="info-grid">
      <div class="info-row">
        <span class="info-key">Tags</span>
        <span class="info-val">{tags_str[:80]}{"..." if len(tags_str)>80 else ""}</span>
      </div>
      <div class="info-row">
        <span class="info-key">Үнэ</span>
        <span class="info-val">{price_str}</span>
      </div>
      <div class="info-row">
        <span class="info-key">Жанр</span>
        <span class="info-val">{genres_str}</span>
      </div>
      <div class="info-row">
        <span class="info-key">Хэл</span>
        <span class="info-val">{langs_str[:60]}{"..." if len(langs_str)>60 else ""}</span>
      </div>
      <div class="info-row" style="grid-column:1/-1;">
        <span class="info-key">Гарсан огноо</span>
        <span class="info-val">{date_str}</span>
      </div>
    </div>
  </div>

</div>

<script>
new Chart(document.getElementById('probChart'), {{
  type: 'bar',
  data: {{
    labels: ['Амжилттай', 'Амжилтгүй'],
    datasets: [{{
      data: [{success_pct}, {fail_pct}],
      backgroundColor: ['#1D9E75', '#D85A30'],
      borderRadius: 6,
      barThickness: 32
    }}]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{
        max: 100,
        ticks: {{
          callback: v => v + '%',
          font: {{ size: 12 }}
        }},
        grid: {{ color: '#f0efe8' }}
      }},
      y: {{
        ticks: {{ font: {{ size: 13 }} }},
        grid: {{ display: false }}
      }}
    }}
  }}
}});
</script>
</body>
</html>"""
    return html


def open_visual_report(results_data):
    html = generate_visual_report(results_data)
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.html',
                                     delete=False, encoding='utf-8')
    tmp.write(html)
    tmp.close()
    path = os.path.abspath(tmp.name)
    print(f"\n📊 Visual report: file://{path}")
    webbrowser.open(f"file://{path}")
    return path


# ─────────────────────────────────────────────
#  INTERACTIVE PREDICTION SYSTEM
# ─────────────────────────────────────────────

def interactive_prediction_system(visual_mode=False):
    """Interactive system — terminal input with optional HTML visual report."""

    train_df = load_and_filter(TRAIN_PATH)
    test_df  = load_and_filter(TEST_PATH)

    top_tags = get_top_tags(train_df, TOP_TAGS_COUNT)

    x_train_tags = encode_tags(train_df, top_tags)
    x_train_num  = build_numeric_features(train_df)
    x_train = pd.concat([x_train_tags.reset_index(drop=True),
                         x_train_num.reset_index(drop=True)], axis=1)
    y_train = train_df["success"]

    model = DecisionTreeClassifier(criterion="gini", max_depth=6,
                                   min_samples_split=20, random_state=42)
    model.fit(x_train, y_train)

    mnb_model = MultinomialNB()
    mnb_model.fit(x_train_tags, y_train)

    gnb_model = GaussianNB()
    gnb_model.fit(x_train_num, y_train)

    print("\n" + "="*60)
    print("🎮 GAME SUCCESS PREDICTOR 🎮")
    print("="*60)
    if visual_mode:
        print("📊 Visual mode ON — browser дээр report нээгдэнэ")
    print("Enter game details below (press Enter after each line)")
    print("Type 'quit' at any prompt to exit")
    print("-"*60)

    while True:
        print("\n" + "🔍 NEW GAME".center(60))
        print("-"*60)

        try:
            print("\n📷 TAGS (comma-separated):")
            print("   Example: Indie, RPG, Story Rich, Adventure")
            tags_input = input("   ➜ Tags: ").strip()
            if tags_input.lower() == 'quit':
                break
            tags = [t.strip() for t in tags_input.split(',') if t.strip()]

            print("\n💰 PRICE (in USD):")
            print("   Example: 19.99, 0.00, 49.99")
            price_input = input("   ➜ Price: $").strip()
            if price_input.lower() == 'quit':
                break
            price = float(price_input)

            print("\n🎭 GENRES (comma-separated):")
            print("   Example: Indie, RPG, Strategy")
            genres_input = input("   ➜ Genres: ").strip()
            if genres_input.lower() == 'quit':
                break
            genres = [g.strip() for g in genres_input.split(',') if g.strip()]

            print("\n🌐 LANGUAGES (comma-separated):")
            print("   Example: English, French, German, Spanish")
            langs_input = input("   ➜ Languages: ").strip()
            if langs_input.lower() == 'quit':
                break
            languages = [l.strip() for l in langs_input.split(',') if l.strip()]

            print("\n📅 RELEASE DATE (YYYY-MM-DD):")
            print("   Example: 2023-06-15")
            release_date = input("   ➜ Release date: ").strip()
            if release_date.lower() == 'quit':
                break

            print("\n" + "="*60)
            print("📋 GAME SUMMARY".center(60))
            print("="*60)
            print(f"   Tags:      {', '.join(tags) if tags else 'None'}")
            print(f"   Price:     ${price:.2f}")
            print(f"   Genres:    {', '.join(genres) if genres else 'None'}")
            print(f"   Languages: {', '.join(languages) if languages else 'None'}")
            print(f"   Released:  {release_date}")

            game_df = pd.DataFrame({
                'tags':           [str(tags)],
                'genres':         [str(genres)],
                'languages':      [str(languages)],
                'price':          [price],
                'release_date':   [release_date],
                'review_summary': ['Unknown']
            })

            x_tags_pred = encode_tags(game_df, top_tags)
            x_num_pred  = build_numeric_features(game_df)
            x_pred = pd.concat([x_tags_pred.reset_index(drop=True),
                                 x_num_pred.reset_index(drop=True)], axis=1)

            pred = model.predict(x_pred)[0]
            prob = model.predict_proba(x_pred)[0]

            print("\n" + "="*60)
            print("🎯 PREDICTION RESULT".center(60))
            print("="*60)

            def confidence_bar(percentage, width=20):
                filled = int(width * percentage / 100)
                bar = '█' * filled + '░' * (width - filled)
                return bar

            if pred == 1:
                print("\n✅ PREDICTION: SUCCESSFUL GAME ✅")
                print(f"\n   Confidence: {prob[1]*100:.1f}%")
                print(f"   {confidence_bar(prob[1]*100)}")
            else:
                print("\n❌ PREDICTION: FAILURE GAME ❌")
                print(f"\n   Confidence: {prob[0]*100:.1f}%")
                print(f"   {confidence_bar(prob[0]*100)}")

            print(f"\n   📊 Success probability: {prob[1]*100:.1f}%")
            print(f"   📊 Failure probability: {prob[0]*100:.1f}%")

            print("\n" + "📊 KEY FACTORS ANALYZED".center(60))
            print("-"*60)

            factors = []
            indie_col    = 'tag_indie'
            action_col   = 'tag_action'
            strategy_col = 'tag_strategy'

            if indie_col in x_tags_pred.columns and x_tags_pred[indie_col].values[0] == 1:
                factors.append(("Has 'Indie' tag", "positive"))
            if action_col in x_tags_pred.columns and x_tags_pred[action_col].values[0] == 1:
                factors.append(("Has 'Action' tag", "positive"))
            if strategy_col in x_tags_pred.columns and x_tags_pred[strategy_col].values[0] == 1:
                factors.append(("Has 'Strategy' tag", "positive"))

            price_val = x_num_pred['price_capped'].values[0]
            if price_val == 0:
                factors.append(("Free game", "positive"))
            elif price_val > 30:
                factors.append(("High price (>$30)", "negative"))
            elif price_val < 10:
                factors.append(("Low price (<$10)", "positive"))

            tag_count_val = x_num_pred['tag_count'].values[0]
            if tag_count_val > 8:
                factors.append(("Many descriptive tags", "positive"))
            elif tag_count_val < 3:
                factors.append(("Few tags (less discovery)", "negative"))

            genre_count_val = x_num_pred['genre_count'].values[0]
            if genre_count_val > 2:
                factors.append(("Multiple genres", "positive"))

            lang_count_val = x_num_pred['lang_count'].values[0]
            if lang_count_val > 5:
                factors.append(("Broad language support", "positive"))
            elif lang_count_val == 1:
                factors.append(("Limited language support", "negative"))

            for label, ftype in factors[:5]:
                icon = "✓" if ftype == "positive" else "⚠️"
                print(f"   {icon} {label}")

            if not factors:
                print("   No strong factors detected")

            print("\n" + "🔄 MODEL COMPARISON".center(60))
            print("-"*60)

            mnb_pred = mnb_model.predict(x_tags_pred)[0]
            mnb_prob = mnb_model.predict_proba(x_tags_pred)[0]

            gnb_pred = gnb_model.predict(x_num_pred)[0]
            gnb_prob = gnb_model.predict_proba(x_num_pred)[0]

            print(f"   Decision Tree:   {'✅ SUCCESS' if pred == 1 else '❌ FAILURE'} ({max(prob)*100:.1f}%)")
            print(f"   Multinomial NB:  {'✅ SUCCESS' if mnb_pred == 1 else '❌ FAILURE'} ({max(mnb_prob)*100:.1f}%)")
            print(f"   Gaussian NB:     {'✅ SUCCESS' if gnb_pred == 1 else '❌ FAILURE'} ({max(gnb_prob)*100:.1f}%)")

            votes = [pred, mnb_pred, gnb_pred]
            majority = 1 if sum(votes) >= 2 else 0

            if majority != pred:
                print(f"\n   💡 Ensemble suggests: {'SUCCESS' if majority == 1 else 'FAILURE'}")

            # ── VISUAL REPORT ──────────────────────────────────
            if visual_mode:
                results_data = {
                    "game_info": {
                        "tags":         tags,
                        "price":        price,
                        "genres":       genres,
                        "languages":    languages,
                        "release_date": release_date,
                    },
                    "pred":     pred,
                    "prob":     prob,
                    "mnb_pred": mnb_pred,
                    "mnb_prob": mnb_prob,
                    "gnb_pred": gnb_pred,
                    "gnb_prob": gnb_prob,
                    "factors":  factors,
                }
                open_visual_report(results_data)

        except ValueError:
            print("\n❌ Error: Invalid number format. Please check your input.")
            print("   Example: Price should be a number like 19.99")
            continue
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("   Please try again with valid input")
            continue

        print("\n" + "-"*60)
        again = input("\n🎮 Predict another game? (yes/no): ").strip().lower()
        if again not in ['yes', 'y']:
            print("\n👋 Thanks for using Game Success Predictor!")
            break


# ─────────────────────────────────────────────
#  MAIN (test set evaluation)
# ─────────────────────────────────────────────

def main():
    train_df = load_and_filter(TRAIN_PATH)
    test_df  = load_and_filter(TEST_PATH)

    print(f"train rows: {len(train_df)}, test rows: {len(test_df)}")
    print(f"train success rate: {train_df['success'].mean():.3f}")
    print(f"test  success rate: {test_df['success'].mean():.3f}")
    print()

    top_tags   = get_top_tags(train_df, TOP_TAGS_COUNT)
    x_train_nb = encode_tags(train_df, top_tags)
    x_test_nb  = encode_tags(test_df, top_tags)
    y_train    = train_df["success"]
    y_test     = test_df["success"]

    mnb_model = run_multinomial_nb(x_train_nb, y_train, x_test_nb, y_test)
    print()

    x_train_num = build_numeric_features(train_df)
    x_test_num  = build_numeric_features(test_df)

    gnb_model = run_gaussian_nb(x_train_num, y_train, x_test_num, y_test)
    print()

    x_train_tree = pd.concat(
        [x_train_nb.reset_index(drop=True),
         x_train_num.reset_index(drop=True)], axis=1)
    x_test_tree = pd.concat(
        [x_test_nb.reset_index(drop=True),
         x_test_num.reset_index(drop=True)], axis=1)

    tree_model = DecisionTreeClassifier(
        criterion="gini", max_depth=6,
        min_samples_split=20, random_state=42)
    tree_model.fit(x_train_tree, y_train)
    y_pred_tree = tree_model.predict(x_test_tree)
    print_results("Decision Tree", y_test, y_pred_tree)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("="*60)
    print("🎮 GAME SUCCESS PREDICTION SYSTEM 🎮")
    print("="*60)
    print("\nSelect mode:")
    print("1. Test on existing test data (original main function)")
    print("2. Interactive prediction (terminal output)")
    print("3. Interactive prediction + Visual HTML report (browser)")
    print("4. Run both test + interactive (terminal output)")

    choice = input("\nEnter your choice (1/2/3/4): ").strip()

    if choice == '1':
        main()
    elif choice == '2':
        interactive_prediction_system(visual_mode=False)
    elif choice == '3':
        interactive_prediction_system(visual_mode=True)
    elif choice == '4':
        print("\n" + "="*60)
        print("RUNNING TEST ON TEST DATA")
        print("="*60)
        main()
        print("\n" + "="*60)
        print("STARTING INTERACTIVE MODE")
        print("="*60)
        interactive_prediction_system(visual_mode=False)
    else:
        print("Invalid choice. Running interactive mode by default.")
        interactive_prediction_system(visual_mode=False)
