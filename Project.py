import ast
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix


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


def main():
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


if __name__ == "__main__":
    main()