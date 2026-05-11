import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import streamlit as st
import shap

# --- 关键修复 1: 解决 ParseException 和渲染问题 ---
import matplotlib
matplotlib.use('Agg') # 使用非交互式后端，更稳定
plt.rcParams['text.usetex'] = False # 禁用 LaTeX 解析，防止报错

st.set_page_config(page_title="Churn Predictor Pro", layout="wide")

# =========================
# 1. 深度缓存：模型与数据
# =========================
@st.cache_resource
def load_model_and_data():
    df = pd.read_csv("ecommerce_customer_churn_dataset.csv")
    
    features = ['Credit_Balance', 'Lifetime_Value', 'Cart_Abandonment_Rate', 
                'Average_Order_Value', 'Days_Since_Last_Purchase']
    
    X = df[features].fillna(df[features].mean())
    y = df['Churned']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    )
    rf.fit(X_train, y_train)

    explainer = shap.TreeExplainer(rf)

    return df, X_train, X_test, y_train, y_test, rf, explainer

# =========================
# 2. 关键优化：缓存全局 SHAP 值 (这是提速的核心)
# =========================
@st.cache_data
def get_global_shap(_explainer, _X):
    # 只取前 100 条数据做全局分析，既能代表整体，速度又快
    sample_X = _X.head(100)
    shap_values = _explainer(sample_X)
    return shap_values, sample_X

# 加载数据、模型和解释器
df, X_train, X_test, y_train, y_test, rf, explainer = load_model_and_data()

# =========================
# 网页界面
# =========================
st.title("Customer Churn Prediction App")

st.caption(
    "Interactive prediction using the top churn-related behavioral features."
)
col1, col2 = st.columns(2)
st.write("### Customer Summary")

st.caption(
    "Customers with high cart abandonment and long inactivity periods tend to show elevated churn risk."
)

with col1:
    credit_balance = st.number_input("Credit Balance", value=2000.0)
    lifetime_value = st.number_input("Lifetime Value", value=1400.0)
    cart_abandonment = st.slider("Cart Abandonment Rate (%)", 0.0, 100.0, 50.0)
with col2:
    average_order = st.number_input("Average Order Value", value=120.0)
    days_since_purchase = st.number_input("Days Since Last Purchase", value=30)

# =========================
# 3. 局部预测 (仅在点击时计算)
# =========================
if st.button("Predict Churn Risk", type="primary"):
    user_data = pd.DataFrame({
        'Credit_Balance': [credit_balance],
        'Lifetime_Value': [lifetime_value],
        'Cart_Abandonment_Rate': [cart_abandonment],
        'Average_Order_Value': [average_order],
        'Days_Since_Last_Purchase': [days_since_purchase]
    })

    prob = rf.predict_proba(user_data)[0][1]
    
    st.divider()
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.subheader("Prediction")
        st.metric("Churn Probability", f"{prob:.2%}")
        if prob > 0.5:
            st.error("High Risk")
        else:
            st.success("Low Risk")

    with res_col2:
        st.write("**Why this score?**")
        # 实时计算单个人的 SHAP，这步很快
        local_shap = explainer(user_data)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        # 增加 try-except 保护，防止特定的数据导致绘图崩溃
        try:
            # 这里的 [0][:, 1] 表示取第一个样本对“流失”类别的贡献
            shap.plots.waterfall(local_shap[0][:, 1], show=False)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"绘图出错: {e}")
        finally:
            plt.close(fig) # 显式关闭，释放内存

# =========================
# 4. 全局分析 (使用缓存数据，秒开)
# =========================
st.divider()
st.subheader("General Model Trends")


# 调用带缓存的 SHAP 计算函数
global_shap_values, sample_X = get_global_shap(explainer, X_train)

tab1, tab2 = st.tabs(["Feature Explanation", "Model Performance"])

with tab1:
    st.subheader("Feature Explanation")
    st.caption(
    "Red indicates higher feature values, blue indicates lower values."
)

    st.caption(
    "Features pushing predictions toward churn appear on the right."
)

    fig_global, ax_global = plt.subplots(figsize=(10, 5))
    shap.summary_plot(
        global_shap_values[:, :, 1],
        sample_X,
        show=False
    )
    plt.tight_layout()
    st.pyplot(fig_global)
    plt.close(fig_global)

with tab2:
    y_prob = rf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    st.metric("AUC Score", f"{auc(fpr, tpr):.4f}")
    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    ax_roc.plot(fpr, tpr, label='ROC Curve')
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    st.pyplot(fig_roc)
    plt.close(fig_roc)

