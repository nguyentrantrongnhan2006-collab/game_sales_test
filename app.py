"""
Ứng dụng Streamlit: Dự đoán doanh số toàn cầu của trò chơi điện tử
====================================================================
Chạy: streamlit run app.py

Cách hoạt động: thay vì load các mô hình sklearn đã được "đóng gói sẵn" (rất dễ vỡ khi
phiên bản scikit-learn giữa lúc lưu và lúc chạy khác nhau — lỗi kiểu "No module named '_loss'"
hay "has no attribute '__pyx_unpickle_...'"), ứng dụng này HUẤN LUYỆN LẠI các mô hình ngay khi
khởi động, từ dữ liệu Train/Test đã được xử lý sẵn (file CSV thuần, không phụ thuộc phiên bản
thư viện). Việc này chỉ mất vài giây và được cache lại (@st.cache_resource) nên chỉ chạy 1 lần
cho mỗi lần khởi động app.

Yêu cầu các file cùng thư mục:
- vgsales_encoding_artifact.joblib  (các map Target Encoding, không chứa object sklearn)
- X_train_encoded.csv, y_train.csv, X_test_encoded.csv, y_test.csv (dữ liệu đã xử lý)
- Cleaned_vgsales_ratings.csv (dữ liệu nguồn, dùng cho tab Dữ liệu & Khai phá)
"""

import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Dự đoán doanh số Game",
    page_icon="🎮",
    layout="wide",
)

# ----------------------------------------------------------------------
# 1. LOAD ENCODING ARTIFACT (chỉ chứa dict/pandas — an toàn với mọi phiên bản sklearn)
# ----------------------------------------------------------------------
@st.cache_resource
def load_encoding_artifact(path="vgsales_encoding_artifact.joblib"):
    return joblib.load(path)


@st.cache_data
def load_source_data(path="Cleaned_vgsales_ratings.csv"):
    return pd.read_csv(path)


@st.cache_resource(show_spinner="Đang huấn luyện mô hình lần đầu (chỉ mất vài giây)...")
def train_models(_encoding_artifact):
    """Huấn luyện 4 mô hình trực tiếp bằng phiên bản scikit-learn đang cài trên máy chủ.
    Tránh hoàn toàn lỗi tương thích phiên bản khi unpickle model đã huấn luyện sẵn ở nơi khác."""
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    X_train = pd.read_csv("X_train_encoded.csv")
    y_train = pd.read_csv("y_train.csv")["y_train"]
    X_test = pd.read_csv("X_test_encoded.csv")
    y_test = pd.read_csv("y_test.csv")["y_test"]

    # Đồng bộ đúng thứ tự cột theo feature_columns đã lưu (đề phòng CSV đổi thứ tự cột)
    feature_columns = _encoding_artifact["feature_columns"]
    X_train = X_train.reindex(columns=feature_columns, fill_value=0)
    X_test = X_test.reindex(columns=feature_columns, fill_value=0)

    dt_model = DecisionTreeRegressor(max_depth=8, min_samples_split=10, min_samples_leaf=5, random_state=42)
    dt_model.fit(X_train, y_train)

    rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=10,
                                      min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    rf_tuned_params = dict(_encoding_artifact["best_rf_tuned_params"])
    rf_model_tuned = RandomForestRegressor(**rf_tuned_params)
    rf_model_tuned.fit(X_train, y_train)

    gb_params = dict(_encoding_artifact["best_gb_params"])
    gb_model = GradientBoostingRegressor(**gb_params)
    gb_model.fit(X_train, y_train)

    models = {
        "Decision Tree": dt_model,
        "Random Forest": rf_model,
        "Random Forest (Tuned)": rf_model_tuned,
        "Gradient Boosting": gb_model,
    }

    # Tính lại metrics thật trên tập Test (thay vì số hardcode) — luôn khớp với model vừa huấn luyện
    metrics = {}
    y_test_orig = np.expm1(y_test)
    baseline_pred = np.full_like(y_test_orig, fill_value=np.expm1(y_train).mean(), dtype=float)
    metrics["Baseline (mean)"] = {
        "MAE": mean_absolute_error(y_test_orig, baseline_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test_orig, baseline_pred)),
        "R2": r2_score(y_test_orig, baseline_pred),
    }
    for name, m in models.items():
        pred = np.expm1(m.predict(X_test))
        metrics[name] = {
            "MAE": mean_absolute_error(y_test_orig, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test_orig, pred)),
            "R2": r2_score(y_test_orig, pred),
        }

    return models, metrics, feature_columns


artifact = None
load_error = None
try:
    artifact = load_encoding_artifact()
except FileNotFoundError:
    load_error = (
        "Không tìm thấy file `vgsales_encoding_artifact.joblib`. "
        "Hãy đặt file này **cùng thư mục** với app.py rồi chạy lại "
        "(kiểm tra bằng lệnh `dir` / `ls` xem file có thật sự nằm đó không)."
    )
except Exception as e:
    load_error = f"Lỗi khi tải file encoding artifact: {e}"

if load_error:
    st.error(load_error)
    st.stop()

if artifact is None:
    raise RuntimeError(
        "Không thể tải encoding artifact và st.stop() không dừng được script. "
        "Hãy kiểm tra lại file vgsales_encoding_artifact.joblib và cách bạn khởi chạy ứng dụng."
    )

try:
    MODELS, MODEL_METRICS, FEATURE_COLUMNS = train_models(artifact)
except FileNotFoundError as e:
    st.error(
        f"Thiếu file dữ liệu huấn luyện ({e}). Cần đủ 4 file: "
        "X_train_encoded.csv, y_train.csv, X_test_encoded.csv, y_test.csv cùng thư mục với app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"Lỗi khi huấn luyện mô hình: {e}")
    st.stop()

SELECTED_FEATURES = artifact["selected_features"]
PUBLISHER_MAP = artifact["publisher_mean_map"]
GP_MAP = artifact["genre_platform_mean_map"]
FRANCHISE_MAP = artifact["franchise_mean_map"]
DEVELOPER_MAP = artifact["developer_mean_map"]
IMPUTE_MEDIANS = artifact["impute_medians"]
GLOBAL_MEAN_SALES = artifact["global_mean_sales"]
PLATFORM_LAUNCH_YEAR = artifact["platform_launch_year"]
GENERATION_MAP = artifact["generation_map"]
KNOWN_PLATFORMS = artifact["known_platforms"]
KNOWN_GENRES = artifact["known_genres"]
KNOWN_RATINGS = [r for r in artifact["known_ratings"] if r != "Unknown"] + ["Unknown"]
KNOWN_PUBLISHERS = artifact["known_publishers"]
KNOWN_DEVELOPERS = artifact["known_developers"]

# Dữ liệu nguồn dùng cho tab "Dữ liệu & Khai phá" — không bắt buộc, thiếu vẫn chạy được các tab khác
try:
    SOURCE_DF = load_source_data()
    source_data_error = None
except Exception as e:
    SOURCE_DF = None
    source_data_error = (
        f"Không tải được file `Cleaned_vgsales_ratings.csv` ({e}). "
        "Tab **Dữ liệu & Khai phá** sẽ tạm thời không khả dụng, các tab khác vẫn dùng bình thường."
    )
KNOWN_RATINGS = [r for r in artifact["known_ratings"] if r != "Unknown"] + ["Unknown"]
KNOWN_PUBLISHERS = artifact["known_publishers"]
KNOWN_DEVELOPERS = artifact["known_developers"]

# Số liệu đánh giá mô hình (đo trên tập Test, xem báo cáo Chương 4)
MODEL_METRICS = {
    "Baseline (mean)":        {"MAE": 0.5543, "RMSE": 1.2159, "R2": -0.0002},
    "Decision Tree":          {"MAE": 0.3998, "RMSE": 0.9356, "R2": 0.4078},
    "Random Forest":          {"MAE": 0.3768, "RMSE": 0.8956, "R2": 0.4574},
    "Random Forest (Tuned)":  {"MAE": 0.3630, "RMSE": 0.9620, "R2": 0.3739},
    "Gradient Boosting":      {"MAE": 0.3676, "RMSE": 0.8721, "R2": 0.4854},
}


# ----------------------------------------------------------------------
# 2. HÀM XỬ LÝ ĐẶC TRƯNG (giống hệt pipeline đã dùng khi huấn luyện)
# ----------------------------------------------------------------------
def extract_franchise(name: str) -> str:
    if not name:
        return "Unknown"
    s = str(name)
    for sep in [":", " - ", "\u2013"]:
        if sep in s:
            s = s.split(sep)[0]
            break
    s = re.sub(r"\s+(I|II|III|IV|V|VI|VII|VIII|IX|X|\d+)$", "", s.strip())
    s = re.sub(r"\s+(Remastered|Deluxe|Edition|HD|Ultimate|Complete|Collection)$", "", s.strip(), flags=re.IGNORECASE)
    return s.strip() or "Unknown"


def is_sequel_from_name(name: str) -> int:
    if not name:
        return 0
    s = str(name).strip()
    return int(bool(re.search(r"\b(II|III|IV|V|VI|VII|VIII|IX|X|[2-9])$", s)))


def life_cycle(years_after_launch: float) -> str:
    if years_after_launch <= 2:
        return "Launch"
    elif years_after_launch <= 5:
        return "Growth"
    elif years_after_launch <= 8:
        return "Mature"
    else:
        return "Late"


def build_feature_row(inputs: dict) -> pd.DataFrame:
    """Nhận dict input thô từ giao diện, trả về 1 dòng DataFrame đã mã hóa sẵn sàng cho model.predict()."""
    platform = inputs["platform"]
    genre = inputs["genre"]
    year = inputs["year"]

    platform_launch_year = PLATFORM_LAUNCH_YEAR.get(platform, year)
    years_after_launch = max(year - platform_launch_year, 0)
    generation = GENERATION_MAP.get(platform, 0)
    life_cycle_val = life_cycle(years_after_launch)

    franchise = extract_franchise(inputs["name"]) if inputs["name"] else "Unknown"
    is_sequel = is_sequel_from_name(inputs["name"]) if inputs["name"] else int(inputs["is_sequel_manual"])

    publisher_mean = PUBLISHER_MAP.get(inputs["publisher"], GLOBAL_MEAN_SALES)
    gp_mean = GP_MAP.get((genre, platform), GLOBAL_MEAN_SALES)
    franchise_mean = FRANCHISE_MAP.get(franchise, GLOBAL_MEAN_SALES)
    developer_mean = DEVELOPER_MAP.get(inputs["developer"], GLOBAL_MEAN_SALES)

    has_critic = int(inputs["critic_score"] is not None)
    has_user = int(inputs["user_score"] is not None)
    critic_score = inputs["critic_score"] if has_critic else IMPUTE_MEDIANS["Critic_Score"]
    critic_count = inputs["critic_count"] if has_critic else IMPUTE_MEDIANS["Critic_Count"]
    user_score_100 = (inputs["user_score"] * 10) if has_user else IMPUTE_MEDIANS["User_Score_100"]
    user_count = inputs["user_count"] if has_user else IMPUTE_MEDIANS["User_Count"]

    row = {
        "Platform": platform,
        "Genre": genre,
        "Rating": inputs["rating"],
        "Publisher_Mean_Sales": publisher_mean,
        "Genre_Platform_Quality": gp_mean,
        "Franchise_Mean_Sales": franchise_mean,
        "Developer_Mean_Sales": developer_mean,
        "Is_Sequel": is_sequel,
        "Critic_Score": critic_score,
        "Critic_Count": critic_count,
        "User_Score_100": user_score_100,
        "User_Count": user_count,
        "Has_Critic_Score": has_critic,
        "Has_User_Score": has_user,
        "Year": year,
        "Platform_Launch_Year": platform_launch_year,
        "Years_After_Launch": years_after_launch,
        "Life_Cycle": life_cycle_val,
        "Generation": generation,
    }
    raw_df = pd.DataFrame([row])[SELECTED_FEATURES]
    encoded = pd.get_dummies(raw_df, drop_first=True)
    encoded = encoded.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return encoded, {
        "franchise": franchise, "years_after_launch": years_after_launch,
        "generation": generation, "life_cycle": life_cycle_val,
        "publisher_mean": publisher_mean, "gp_mean": gp_mean,
        "franchise_mean": franchise_mean, "developer_mean": developer_mean,
    }


# ----------------------------------------------------------------------
# 3. GIAO DIỆN
# ----------------------------------------------------------------------
st.title("🎮 Dự đoán doanh số toàn cầu của trò chơi điện tử")
st.caption(
    "Dựa trên bộ dữ liệu Video Game Sales with Ratings (Kaggle). "
    "Nhập thông tin một tựa game để dự đoán doanh số toàn cầu (triệu bản)."
)

tab_predict, tab_eda, tab_performance, tab_about, tab_source = st.tabs(
    ["🔮 Dự đoán", "📁 Dữ liệu & Khai phá", "📊 Hiệu suất mô hình", "ℹ️ Giới thiệu", "💻 Source Code"]
)

# ================= TAB 1: DỰ ĐOÁN =================
with tab_predict:
    col_form, col_result = st.columns([1.3, 1])

    with col_form:
        st.subheader("Thông tin trò chơi")

        model_name = st.selectbox(
            "Chọn mô hình dự đoán",
            list(MODELS.keys()),
            index=list(MODELS.keys()).index("Gradient Boosting"),
            help="Gradient Boosting đạt R² cao nhất trên tập Test trong quá trình đánh giá.",
        )

        name = st.text_input(
            "Tên game (tùy chọn — dùng để tự nhận diện Franchise/Sequel)",
            placeholder="Ví dụ: Call of Duty: Black Ops III",
        )

        c1, c2 = st.columns(2)
        with c1:
            platform = st.selectbox("Nền tảng (Platform)", KNOWN_PLATFORMS, index=KNOWN_PLATFORMS.index("PS4") if "PS4" in KNOWN_PLATFORMS else 0)
            genre = st.selectbox("Thể loại (Genre)", KNOWN_GENRES)
            rating = st.selectbox("Xếp hạng ESRB (Rating)", KNOWN_RATINGS)
            year = st.number_input("Năm phát hành", min_value=1980, max_value=2030, value=2016, step=1)
        with c2:
            publisher = st.selectbox("Nhà phát hành (Publisher)", ["(Chưa xác định)"] + KNOWN_PUBLISHERS)
            developer = st.selectbox("Nhà phát triển (Developer)", ["(Chưa xác định)"] + KNOWN_DEVELOPERS)
            is_sequel_manual = st.checkbox("Đây là phần hậu truyện (sequel)?", value=False, disabled=bool(name))

        st.markdown("**Điểm đánh giá (bỏ trống nếu chưa có — hệ thống sẽ tự điền giá trị trung vị)**")
        c3, c4 = st.columns(2)
        with c3:
            has_critic_input = st.checkbox("Có điểm Critic Score?", value=True)
            critic_score = st.slider("Critic Score (0-100)", 0, 100, 75, disabled=not has_critic_input)
            critic_count = st.number_input("Số lượng Critic đánh giá", min_value=0, max_value=200, value=20, disabled=not has_critic_input)
        with c4:
            has_user_input = st.checkbox("Có điểm User Score?", value=True)
            user_score = st.slider("User Score (0-10)", 0.0, 10.0, 7.5, step=0.1, disabled=not has_user_input)
            user_count = st.number_input("Số lượng User đánh giá", min_value=0, max_value=20000, value=100, disabled=not has_user_input)

        predict_btn = st.button("🚀 Dự đoán doanh số", type="primary", use_container_width=True)

    with col_result:
        st.subheader("Kết quả dự đoán")
        if predict_btn:
            inputs = dict(
                name=name.strip(),
                platform=platform,
                genre=genre,
                rating=rating,
                publisher=publisher if publisher != "(Chưa xác định)" else "Unknown",
                developer=developer if developer != "(Chưa xác định)" else "Unknown",
                is_sequel_manual=is_sequel_manual,
                year=int(year),
                critic_score=float(critic_score) if has_critic_input else None,
                critic_count=float(critic_count) if has_critic_input else None,
                user_score=float(user_score) if has_user_input else None,
                user_count=float(user_count) if has_user_input else None,
            )
            X_row, debug_info = build_feature_row(inputs)
            model = MODELS[model_name]
            pred_log = model.predict(X_row)[0]
            pred_sales = float(np.expm1(pred_log))

            st.metric("Doanh số toàn cầu dự đoán", f"{pred_sales:.2f} triệu bản")

            launch_year_for_platform = PLATFORM_LAUNCH_YEAR.get(platform, year)
            if year < launch_year_for_platform:
                st.warning(
                    f"Năm phát hành ({year}) sớm hơn năm ra mắt nền tảng {platform} "
                    f"({launch_year_for_platform}). Đã tự điều chỉnh Years_After_Launch = 0."
                )

            with st.expander("Chi tiết đặc trưng đã suy ra"):
                st.write(
                    {
                        "Franchise (suy ra từ tên)": debug_info["franchise"],
                        "Vòng đời sản phẩm (Life_Cycle)": debug_info["life_cycle"],
                        "Số năm sau khi nền tảng ra mắt": debug_info["years_after_launch"],
                        "Thế hệ console (Generation)": debug_info["generation"],
                        "Publisher_Mean_Sales": round(debug_info["publisher_mean"], 3),
                        "Genre_Platform_Quality": round(debug_info["gp_mean"], 3),
                        "Franchise_Mean_Sales": round(debug_info["franchise_mean"], 3),
                        "Developer_Mean_Sales": round(debug_info["developer_mean"], 3),
                    }
                )
        else:
            st.info("Điền thông tin bên trái rồi bấm **Dự đoán doanh số** để xem kết quả.")

# ================= TAB 2: DỮ LIỆU & KHAI PHÁ =================
with tab_eda:
    if source_data_error:
        st.warning(source_data_error)
    else:
        df = SOURCE_DF.copy()

        st.subheader("Tổng quan dữ liệu nguồn")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Số dòng", f"{df.shape[0]:,}")
        c2.metric("Số cột", df.shape[1])
        c3.metric("Số Publisher", df["Publisher"].nunique())
        c4.metric("Khoảng năm", f"{int(df['Year'].min())}–{int(df['Year'].max())}")

        with st.expander("Xem mẫu dữ liệu thô"):
            st.dataframe(df.head(50), use_container_width=True)

        with st.expander("Thống kê mô tả các cột số"):
            st.dataframe(df.describe(), use_container_width=True)

        st.divider()
        st.subheader("Bộ lọc khai phá")
        f1, f2, f3 = st.columns(3)
        with f1:
            genre_filter = st.multiselect("Thể loại (Genre)", sorted(df["Genre"].unique()), default=[])
        with f2:
            platform_filter = st.multiselect("Nền tảng (Platform)", sorted(df["Platform"].unique()), default=[])
        with f3:
            year_min, year_max = int(df["Year"].min()), int(df["Year"].max())
            year_range = st.slider("Khoảng năm phát hành", year_min, year_max, (year_min, year_max))

        filtered = df.copy()
        if genre_filter:
            filtered = filtered[filtered["Genre"].isin(genre_filter)]
        if platform_filter:
            filtered = filtered[filtered["Platform"].isin(platform_filter)]
        filtered = filtered[(filtered["Year"] >= year_range[0]) & (filtered["Year"] <= year_range[1])]
        st.caption(f"Đang khai phá trên {filtered.shape[0]:,} / {df.shape[0]:,} dòng sau khi lọc.")

        st.divider()
        colA, colB = st.columns(2)

        with colA:
            st.markdown("**Tổng doanh số theo thể loại (Genre)**")
            genre_sales = filtered.groupby("Genre")["Global_Sales"].sum().sort_values(ascending=False)
            st.bar_chart(genre_sales)

        with colB:
            st.markdown("**Tổng doanh số theo thế hệ console (Generation)**")
            gen_sales = filtered.groupby("Generation")["Global_Sales"].sum().sort_index()
            st.bar_chart(gen_sales)

        colC, colD = st.columns(2)
        with colC:
            st.markdown("**Doanh số trung bình theo vòng đời sản phẩm (Life_Cycle)**")
            order = ["Launch", "Growth", "Mature", "Late"]
            lc_sales = filtered.groupby("Life_Cycle")["Global_Sales"].mean().reindex(order)
            st.bar_chart(lc_sales)

        with colD:
            st.markdown("**Doanh số trung bình theo xếp hạng ESRB (Rating)**")
            rating_sales = (
                filtered[filtered["Rating"] != "Unknown"]
                .groupby("Rating")["Global_Sales"].mean()
                .sort_values(ascending=False)
            )
            st.bar_chart(rating_sales)

        st.markdown("**Số lượng game phát hành theo thể loại qua các năm**")
        genre_year = filtered.groupby(["Year", "Genre"]).size().reset_index(name="Số lượng")
        genre_year_pivot = genre_year.pivot(index="Year", columns="Genre", values="Số lượng").fillna(0)
        st.line_chart(genre_year_pivot)

        st.markdown("**Top 10 Publisher theo tổng doanh số** (trong phạm vi đã lọc)")
        top_pub = filtered.groupby("Publisher")["Global_Sales"].sum().sort_values(ascending=False).head(10)
        st.bar_chart(top_pub)

        st.divider()
        st.subheader("Mối quan hệ giữa điểm đánh giá và doanh số")
        score_type = st.radio("Chọn loại điểm đánh giá", ["Critic Score", "User Score"], horizontal=True)
        has_col = "Has_Critic_Score" if score_type == "Critic Score" else "Has_User_Score"
        score_col = "Critic_Score" if score_type == "Critic Score" else "User_Score_100"
        scored = filtered[filtered[has_col] == 1]
        if len(scored) > 0:
            corr = scored[[score_col, "Global_Sales"]].corr().iloc[0, 1]
            st.caption(f"Hệ số tương quan {score_type} — Global Sales: **{corr:.3f}** (trên {len(scored):,} game có điểm đánh giá)")
            st.scatter_chart(scored, x=score_col, y="Global_Sales", size=None, height=400)
        else:
            st.info("Không có dữ liệu điểm đánh giá trong phạm vi đã lọc.")

        st.divider()
        st.subheader("Nhận xét nhanh")
        st.markdown(
            """
- Console thế hệ thứ 7 (Xbox 360, PS3, Wii) và các thể loại **Action, Sports** đóng góp phần lớn doanh số toàn ngành.
- Doanh số trung bình mỗi game cao nhất ở giai đoạn **Launch** (mới ra mắt nền tảng) và giảm dần theo vòng đời console.
- **Critic Score** có tương quan với doanh số rõ ràng hơn **User Score** — gợi ý chất lượng chuyên môn ảnh hưởng đến thương mại nhiều hơn đánh giá đại chúng.
- Game xếp hạng **M (Mature)** thường có doanh số trung bình cao nhất trong các nhóm ESRB.

*Dùng bộ lọc phía trên để tự khám phá theo Genre/Platform/khoảng năm bạn quan tâm.*
            """
        )

# ================= TAB 3: HIỆU SUẤT MÔ HÌNH =================
with tab_performance:
    st.subheader("So sánh các mô hình trên tập Test")
    metrics_df = pd.DataFrame(MODEL_METRICS).T.reset_index().rename(columns={"index": "Model"})
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.bar_chart(metrics_df.set_index("Model")["MAE"])
    c2.bar_chart(metrics_df.set_index("Model")["RMSE"])
    c3.bar_chart(metrics_df.set_index("Model")["R2"])

    st.subheader("Feature Importance")
    chosen_for_importance = st.selectbox(
        "Chọn mô hình để xem mức độ quan trọng của đặc trưng",
        [m for m in MODELS.keys()],
        key="importance_model",
    )
    imp_model = MODELS[chosen_for_importance]
    imp_df = (
        pd.DataFrame({"Feature": FEATURE_COLUMNS, "Importance": imp_model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .head(15)
        .set_index("Feature")
    )
    st.bar_chart(imp_df)

# ================= TAB 4: GIỚI THIỆU =================
with tab_about:
    st.markdown(
        """
### Về ứng dụng này

Ứng dụng dự đoán **doanh số toàn cầu (Global Sales)** của một trò chơi điện tử dựa trên:
- Thể loại, nền tảng, nhà phát hành, nhà phát triển, xếp hạng độ tuổi
- Điểm đánh giá chuyên môn (Critic Score) và người dùng (User Score)
- Các đặc trưng được kỹ thuật hóa: `Publisher_Mean_Sales`, `Genre_Platform_Quality`,
  `Franchise_Mean_Sales`, `Developer_Mean_Sales` (Target Encoding có Bayesian Smoothing),
  `Years_After_Launch`, `Life_Cycle`, `Generation`.

**Dữ liệu huấn luyện:** Video Game Sales with Ratings (Kaggle),
`Video_Games_Sales_as_at_22_Dec_2016.csv`.

**Mô hình:** Decision Tree, Random Forest (mặc định và đã tối ưu bằng RandomizedSearchCV),
Gradient Boosting Regressor — huấn luyện trên biến mục tiêu `log1p(Global_Sales)` để giảm
ảnh hưởng của phân phối lệch phải mạnh.

**Lưu ý:** Đây là mô hình mang tính tham khảo, huấn luyện trên dữ liệu lịch sử đến cuối 2016.
Doanh số dự đoán không tính đến các yếu tố marketing, cạnh tranh thị trường thời điểm hiện tại,
hoặc các game chưa từng xuất hiện trong dữ liệu huấn luyện.
        """
    )

# ================= TAB 5: SOURCE CODE =================
with tab_source:
    st.subheader("Mã nguồn ứng dụng (app.py)")

    GITHUB_REPO_URL = "https://github.com/nguyentrantrongnhan2006-collab?tab=repositories"  # TODO: dán link repo GitHub của bạn vào đây, ví dụ: "https://github.com/<user>/<repo>"
    if GITHUB_REPO_URL:
        st.link_button("🔗 Xem trên GitHub", GITHUB_REPO_URL, use_container_width=False)
    else:
        st.info(
            "💡 Bạn có thể dán link GitHub repo vào biến `GITHUB_REPO_URL` ở đầu tab này "
            "để hiện nút bấm mở trực tiếp trên GitHub."
        )

    try:
        with open(__file__, "r", encoding="utf-8") as f:
            source_code = f.read()
    except Exception as e:
        source_code = None
        st.error(f"Không đọc được mã nguồn: {e}")

    if source_code:
        st.caption(f"Tổng cộng {len(source_code.splitlines())} dòng code.")
        st.download_button(
            "⬇️ Tải app.py",
            data=source_code,
            file_name="app.py",
            mime="text/x-python",
        )
        with st.expander("📄 Xem toàn bộ mã nguồn", expanded=True):
            st.code(source_code, language="python", line_numbers=True)
