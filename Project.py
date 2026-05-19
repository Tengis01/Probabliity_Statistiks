import ast
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier


TRAIN_PATH = "train.csv"
TEST_PATH  = "test.csv"

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

# tag-уудаас хамгийн өргөн тархсан top n-г сонгоно
TOP_TAGS_COUNT = 50


def load_and_filter(path):
    df = pd.read_csv(path)
    valid = SUCCESS_LABELS | FAILURE_LABELS
    df = df[df["review_summary"].isin(valid)].copy()
    df["success"] = df["review_summary"].isin(SUCCESS_LABELS).astype(int)
    return df


def parse_list_col(series):
    # JSON list string-ийг Python list болгон parse хийнэ
    def safe_parse(val):
        try:
            return ast.literal_eval(val)
        except Exception:
            return []
    return series.apply(safe_parse)


def get_top_tags(train_df, n):
    # train дээрх хамгийн их давтагдсан n tag-ийг буцаана
    tag_counts = {}
    for tag_list in parse_list_col(train_df["tags"]):
        for tag in tag_list:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    sorted_tags = sorted(tag_counts, key=tag_counts.get, reverse=True)
    return sorted_tags[:n]


def encode_tags(df, top_tags):
    # top_tags тус бүр binary feature болгоно (multi-hot)
    tag_lists = parse_list_col(df["tags"])
    tag_set_list = [set(lst) for lst in tag_lists]
    encoded = {}
    for tag in top_tags:
        col_name = f"tag_{tag.lower().replace(' ', '_')}"
        encoded[col_name] = [1 if tag in tag_set else 0 for tag_set in tag_set_list]
    return pd.DataFrame(encoded, index=df.index)


def build_numeric_features(df):
    # тоон feature-үүдийг гарган авна
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

    # success болон failure тус бүрийн зөв таарсан хувь
    success_total = tp + fn
    failure_total = tn + fp
    success_correct_pct = tp / success_total * 100 if success_total > 0 else 0
    failure_correct_pct = tn / failure_total * 100 if failure_total > 0 else 0

    print(f"загвар: {model_name}")
    print(f"нийт {total} тоглоомоос {correct} зөв таарсан ({acc*100:.1f}%)")
    print()
    print(f"  амжилттай тоглоом:  {success_total}-аас {tp} зөв  ({success_correct_pct:.1f}%)")
    print(f"  амжилтгүй тоглоом:  {failure_total}-аас {tn} зөв  ({failure_correct_pct:.1f}%)")
    print()
    print(f"  алдаа 1 - амжилттайг амжилтгүй гэж таасан: {fn}")
    print(f"  алдаа 2 - амжилтгүйг амжилттай гэж таасан: {fp}")


def run_multinomial_nb(x_train, y_train, x_test, y_test):
    # tag-уудыг multi-hot encoding хийсэн feature дээр MultinomialNB ажиллуулна.
    # MultinomialNB нь тоологдох утга (0 эсвэл 1, count) дээр ажилладаг —
    # spam detection дээр үгийн давтамж ашигладагтай ижил зарчим.
    # энд "Indie tag байна уу, үгүй юу" гэсэн 50 асуултын хариултаар
    # тоглоом амжилттай эсэхийг таарварлана.
    model = MultinomialNB()

    # train өгөгдлөөр P(tag | success) болон P(tag | failure) магадлалуудыг тооцоолно
    model.fit(x_train, y_train)

    # test өгөгдлийн tag-уудаар Bayes теоремоор класс таарварлана:
    # P(success | tags) ∝ P(tags | success) * P(success)
    y_pred = model.predict(x_test)

    print_results("Naive Bayes - MultinomialNB (tag feature)", y_test, y_pred)
    return model


def run_gaussian_nb(x_train, y_train, x_test, y_test):
    # price, tag_count, lang_count, release_year зэрэг тоон feature дээр GaussianNB ажиллуулна.
    # GaussianNB нь feature бүрийн утга normal (Gaussian) distribution дагана гэж үзэж
    # train үед класс тус бүрийн дундаж болон стандарт хазайлтыг хадгална.
    # жишээ нь: success тоглоомын price-ийн дундаж болон failure-ийнхтэй харьцуулж
    # шинэ тоглоомын price аль бүлэгт илүү магадлалтай орох вэ гэдгийг тооцдог.
    model = GaussianNB()

    # train өгөгдлөөр feature бүрийн дундаж, стандарт хазайлтыг класс тус бүрд тооцно
    model.fit(x_train, y_train)

    # test өгөгдлийн утга бүрийг gaussian_pdf(x, mean, std) томьёогоор магадлал болгон
    # хөрвүүлж P(success | features) > P(failure | features) эсэхийг шалгана
    y_pred = model.predict(x_test)

    print_results("Naive Bayes - GaussianNB (numeric feature)", y_test, y_pred)
    return model


def interactive_prediction_system():
    """Full interactive system with trained models - Line by line input"""
    
    train_df = load_and_filter(TRAIN_PATH)
    test_df = load_and_filter(TEST_PATH)
    
    top_tags = get_top_tags(train_df, TOP_TAGS_COUNT)
    
    # Prepare training data
    x_train_tags = encode_tags(train_df, top_tags)
    x_train_num = build_numeric_features(train_df)
    x_train = pd.concat([x_train_tags.reset_index(drop=True),
                         x_train_num.reset_index(drop=True)], axis=1)
    y_train = train_df["success"]
    
    # Train model
    model = DecisionTreeClassifier(criterion="gini", max_depth=6, 
                                   min_samples_split=20, random_state=42)
    model.fit(x_train, y_train)
    
    print("\n" + "="*60)
    print("🎮 GAME SUCCESS PREDICTOR 🎮")
    print("="*60)
    print("Enter game details below (press Enter after each line)")
    print("Type 'quit' at any prompt to exit")
    print("-"*60)
    
    while True:
        print("\n" + "🔍 NEW GAME".center(60))
        print("-"*60)
        
        try:
            # Tags input
            print("\n📷 TAGS (comma-separated):")
            print("   Example: Indie, RPG, Story Rich, Adventure")
            tags_input = input("   ➜ Tags: ").strip()
            if tags_input.lower() == 'quit':
                break
            tags = [t.strip() for t in tags_input.split(',') if t.strip()]
            
            # Price input
            print("\n💰 PRICE (in USD):")
            print("   Example: 19.99, 0.00, 49.99")
            price_input = input("   ➜ Price: $").strip()
            if price_input.lower() == 'quit':
                break
            price = float(price_input)
            
            # Genres input
            print("\n🎭 GENRES (comma-separated):")
            print("   Example: Indie, RPG, Strategy")
            genres_input = input("   ➜ Genres: ").strip()
            if genres_input.lower() == 'quit':
                break
            genres = [g.strip() for g in genres_input.split(',') if g.strip()]
            
            # Languages input
            print("\n🌐 LANGUAGES (comma-separated):")
            print("   Example: English, French, German, Spanish")
            langs_input = input("   ➜ Languages: ").strip()
            if langs_input.lower() == 'quit':
                break
            languages = [l.strip() for l in langs_input.split(',') if l.strip()]
            
            # Release date input
            print("\n📅 RELEASE DATE (YYYY-MM-DD):")
            print("   Example: 2023-06-15")
            release_date = input("   ➜ Release date: ").strip()
            if release_date.lower() == 'quit':
                break
            
            # Show summary before prediction
            print("\n" + "="*60)
            print("📋 GAME SUMMARY".center(60))
            print("="*60)
            print(f"   Tags:      {', '.join(tags)}")
            print(f"   Price:     ${price:.2f}")
            print(f"   Genres:    {', '.join(genres)}")
            print(f"   Languages: {', '.join(languages)}")
            print(f"   Released:  {release_date}")
            
            # Create DataFrame
            game_df = pd.DataFrame({
                'tags': [str(tags)],
                'genres': [str(genres)],
                'languages': [str(languages)],
                'price': [price],
                'release_date': [release_date],
                'review_summary': ['Unknown']
            })
            
            # Process features
            x_tags_pred = encode_tags(game_df, top_tags)
            x_num_pred = build_numeric_features(game_df)
            x_pred = pd.concat([x_tags_pred.reset_index(drop=True),
                               x_num_pred.reset_index(drop=True)], axis=1)
            
            # Predict
            pred = model.predict(x_pred)[0]
            prob = model.predict_proba(x_pred)[0]
            
            # Display results with visual feedback
            print("\n" + "="*60)
            print("🎯 PREDICTION RESULT".center(60))
            print("="*60)
            
            # Create progress bar for confidence
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
            
            # Feature importance feedback
            print("\n" + "📊 KEY FACTORS ANALYZED".center(60))
            print("-"*60)
            
            # Check various features
            factors = []
            
            if 'tag_indie' in x_tags_pred.columns and x_tags_pred['tag_indie'].values[0] == 1:
                factors.append(("✓ Has 'Indie' tag", "positive"))
            if 'tag_action' in x_tags_pred.columns and x_tags_pred['tag_action'].values[0] == 1:
                factors.append(("✓ Has 'Action' tag", "positive"))
            if 'tag_strategy' in x_tags_pred.columns and x_tags_pred['tag_strategy'].values[0] == 1:
                factors.append(("✓ Has 'Strategy' tag", "positive"))
            
            if x_num_pred['price_capped'].values[0] == 0:
                factors.append(("✓ Free game", "positive"))
            elif x_num_pred['price_capped'].values[0] > 30:
                factors.append(("⚠️ High price (>$30)", "negative"))
            elif x_num_pred['price_capped'].values[0] < 10:
                factors.append(("✓ Low price (<$10)", "positive"))
                
            if x_num_pred['tag_count'].values[0] > 8:
                factors.append(("✓ Many descriptive tags", "positive"))
            elif x_num_pred['tag_count'].values[0] < 3:
                factors.append(("⚠️ Few tags (less discovery)", "negative"))
                
            if x_num_pred['genre_count'].values[0] > 2:
                factors.append(("✓ Multiple genres", "positive"))
                
            if x_num_pred['lang_count'].values[0] > 5:
                factors.append(("✓ Broad language support", "positive"))
            elif x_num_pred['lang_count'].values[0] == 1:
                factors.append(("⚠️ Limited language support", "negative"))
            
            if factors:
                for factor, _ in factors[:5]:  
                    print(f"   {factor}")
            else:
                print("   No strong factors detected")
            
            # Alternative prediction using other models for comparison
            print("\n" + "🔄 MODEL COMPARISON".center(60))
            print("-"*60)
            
            # Also predict with Naive Bayes models for comparison
            mnb_model = MultinomialNB()
            mnb_model.fit(x_train_tags, y_train)
            mnb_pred = mnb_model.predict(x_tags_pred)[0]
            mnb_prob = mnb_model.predict_proba(x_tags_pred)[0]
            
            gnb_model = GaussianNB()
            gnb_model.fit(x_train_num, y_train)
            gnb_pred = gnb_model.predict(x_num_pred)[0]
            gnb_prob = gnb_model.predict_proba(x_num_pred)[0]
            
            print(f"   Decision Tree:   {'✅ SUCCESS' if pred == 1 else '❌ FAILURE'} ({max(prob)*100:.1f}%)")
            print(f"   Multinomial NB:  {'✅ SUCCESS' if mnb_pred == 1 else '❌ FAILURE'} ({max(mnb_prob)*100:.1f}%)")
            print(f"   Gaussian NB:     {'✅ SUCCESS' if gnb_pred == 1 else '❌ FAILURE'} ({max(gnb_prob)*100:.1f}%)")
            
            # Final recommendation based on majority vote
            votes = [pred, mnb_pred, gnb_pred]
            majority = 1 if sum(votes) >= 2 else 0
            
            if majority != pred:
                print(f"\n   💡 Ensemble suggests: {'SUCCESS' if majority == 1 else 'FAILURE'}")
            
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
        
        # Ask for another prediction
        print("\n" + "-"*60)
        again = input("\n🎮 Predict another game? (yes/no): ").strip().lower()
        if again not in ['yes', 'y']:
            print("\n👋 Thanks for using Game Success Predictor!")
            break


def main():
    """Original main function for testing on test data"""
    train_df = load_and_filter(TRAIN_PATH)
    test_df  = load_and_filter(TEST_PATH)

    print(f"train rows: {len(train_df)}, test rows: {len(test_df)}")
    print(f"train success rate: {train_df['success'].mean():.3f}")
    print(f"test  success rate: {test_df['success'].mean():.3f}")
    print()

    # Naive Bayes - MultinomialNB
    top_tags   = get_top_tags(train_df, TOP_TAGS_COUNT)
    x_train_nb = encode_tags(train_df, top_tags)
    x_test_nb  = encode_tags(test_df, top_tags)
    y_train    = train_df["success"]
    y_test     = test_df["success"]

    mnb_model = run_multinomial_nb(x_train_nb, y_train, x_test_nb, y_test)
    print()

    # Naive Bayes - GaussianNB
    x_train_num = build_numeric_features(train_df)
    x_test_num  = build_numeric_features(test_df)

    gnb_model = run_gaussian_nb(x_train_num, y_train, x_test_num, y_test)
    print()

    # Decision tree
    x_train_tree = pd.concat(
        [x_train_nb.reset_index(drop=True),
         x_train_num.reset_index(drop=True)],
        axis=1
    )

    x_test_tree = pd.concat(
        [x_test_nb.reset_index(drop=True),
         x_test_num.reset_index(drop=True)],
        axis=1
    )

    tree_model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=6,
        min_samples_split=20,
        random_state=42
    )

    tree_model.fit(x_train_tree, y_train)

    y_pred_tree = tree_model.predict(x_test_tree)

    print_results("Decision Tree", y_test, y_pred_tree)


if __name__ == "__main__":
    # Ask user which mode to run
    print("="*60)
    print("🎮 GAME SUCCESS PREDICTION SYSTEM 🎮")
    print("="*60)
    print("\nSelect mode:")
    print("1. Test on existing test data (original main function)")
    print("2. Interactive prediction (enter your own game details)")
    print("3. Run both")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == '1':
        main()
    elif choice == '2':
        interactive_prediction_system()
    elif choice == '3':
        print("\n" + "="*60)
        print("RUNNING TEST ON TEST DATA")
        print("="*60)
        main()
        print("\n" + "="*60)
        print("STARTING INTERACTIVE MODE")
        print("="*60)
        interactive_prediction_system()
    else:
        print("Invalid choice. Running interactive mode by default.")
        interactive_prediction_system()
