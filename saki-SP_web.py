import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 在导入pyplot前设置
import matplotlib.pyplot as plt
from PIL import Image

# 设置页面配置
st.set_page_config(page_title="SA-AKI Subphenotype Prediction Model", layout="wide")

# 标题栏容器
header_container = st.container()
with header_container:
    cols = st.columns([0.2, 0.8])
    with cols[0]:
        try:
            logo = Image.open("东华医院图标.png")
            st.image(logo, use_column_width=True)
        except FileNotFoundError:
            st.write("Logo not found")
    with cols[1]:
        st.title("SA-AKI Subphenotype Prediction Model")
        st.markdown("""
            <div style='border-left: 5px solid #1A5276; padding-left: 15px;'>
            <h4 style='color: #2E86C1;'>Clinical Decision Support System</h4>
            <p style='font-size:16px;'>Dongguan Tungwah Hospital</p>
            <p style='font-size:16px;'>Dongguan Songshan Lake Tungwah Hospital</p>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# 尝试加载4个模型
try:
    model1 = joblib.load('1224sakiSPI_XGB_model1.pkl')
    model2 = joblib.load('1224sakiSPII_XGB_model2.pkl')
    model3 = joblib.load('1224sakiSPIII_XGB_model3.pkl')
    model4 = joblib.load('1224sakiSPIV_XGB_model4.pkl')
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# 定义特征参数
feature_ranges = {
    'Heart rate': {"type": "numerical", "min": 20, "max": 250, "default": 80, "unit": "bpm"},
    'MAP': {"type": "numerical", "min": 30, "max": 200, "default": 90, "unit": "mmHg"},
    'Resp rate': {"type": "numerical", "min": 5, "max": 60, "default": 20, "unit": "bpm"},
    'Temperature': {"type": "numerical", "min": 30, "max": 45, "default": 36.5, "unit": "°C"},
    'spo2': {"type": "numerical", "min": 50, "max": 100, "default": 100, "unit": "%"},
    'WBC': {"type": "numerical", "min": 0.1, "max": 100, "default": 7, "unit": "x10^9/L"},
    'Bicarbonate': {"type": "numerical", "min": 5, "max": 50, "default": 24, "unit": "mmol/L"},
    'Calcium': {"type": "numerical", "min": 4, "max": 20, "default": 8, "unit": "mg/dL"},
    'Creatinine': {"type": "numerical", "min": 0.1, "max": 20, "default": 1, "unit": "mg/dL"},
    'Glucose': {"type": "numerical", "min": 20, "max": 1500, "default": 90, "unit": "mg/dL"},
    'Sodium': {"type": "numerical", "min": 110, "max": 170, "default": 140, "unit": "mmol/L"},
    'Potassium': {"type": "numerical", "min": 2, "max": 9, "default": 4.0, "unit": "mmol/L"},
    'INR': {"type": "numerical", "min": 0.5, "max": 20, "default": 1.0},
    'PTT': {"type": "numerical", "min": 5, "max": 150, "default": 30, "unit": "seconds"},
    'Hemoglobin': {"type": "numerical", "min": 3, "max": 25, "default": 14, "unit": "g/dL"},
    'MCHC': {"type": "numerical", "min": 25, "max": 40, "default": 30, "unit": "g/dL"},
    'MCV': {"type": "numerical", "min": 50, "max": 130, "default": 90, "unit": "fL"},
    'Platelet': {"type": "numerical", "min": 1, "max": 2000, "default": 250, "unit": "x10^9/L"},
    'Urineoutput 6hr': {"type": "numerical", "min": 0, "default": 500, "unit": "mL"}
}

# 定义模型训练时的特征顺序
feature_order = [
    'Heart rate', 'MAP', 'Resp rate', 'Temperature', 'spo2', 'WBC', 
    'Bicarbonate', 'Calcium', 'Creatinine', 'Glucose', 'Sodium', 
    'Potassium', 'INR', 'PTT', 'Hemoglobin', 'MCHC', 'MCV', 'Platelet', 'Urineoutput 6hr'
]

# 简化后的亚型描述字典
sp_descriptions = {
    "Subphenotype I": {
        "color": "#90EE90",
        "prognosis": "In-hospital mortality 16.4%"
    },
    "Subphenotype II": {
        "color": "#87CEEB",
        "prognosis": "In-hospital mortality 14.3%"
    },
    "Subphenotype III": {
        "color": "#FFD700",
        "prognosis": "In-hospital mortality 21.6%"
    },
    "Subphenotype IV": {
        "color": "#FF6B6B",
        "prognosis": "In-hospital mortality 34.2%"
    }
}

# 页面样式
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .prediction-box {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 20px;
        margin-top: 20px;
        background-color: #f9f9f9;
    }
    .stage-card {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 12px;
        background-color: #fff;
        border-left: 5px solid #ccc;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .probability-bar {
        width: 100%;
        height: 30px;
        background-color: #e0e0e0;
        border-radius: 5px;
        overflow: hidden;
        margin-top: 8px;
    }
    .probability-fill {
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# 创建两列布局
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Patient Clinical Parameters")
    st.markdown("Enter values the first recorded clinical variables within the ±24-hour window of SA-AKI diagnosis:")
    
    input_values = {}
    
    # 分组显示参数（根据论文中的变量分类）
    groups = {
        "Vital Signs": ['Heart rate', 'MAP', 'Resp rate', 'Temperature', 'spo2'],
        "Renal Function": ['Creatinine','Urineoutput 6hr'],
        "Inflammation & Hematology": ['WBC', 'Hemoglobin', 'MCHC', 'MCV', 'Platelet'],
        "Metabolic & Electrolytes": ['Bicarbonate', 'Calcium', 'Glucose', 'Sodium', 'Potassium'],
        "Coagulation": ['INR', 'PTT']
    }
    
    for group_name, features in groups.items():
        with st.expander(f"📋 {group_name}", expanded=(group_name == "Vital Signs")):
            for feature in features:
                properties = feature_ranges[feature]
                if properties["type"] == "numerical":
                    value = st.number_input(
                        label=f"{feature} ({properties['unit']})",
                        min_value=float(properties["min"]),
                        max_value=float(properties.get("max", 100000)),
                        value=float(properties["default"]),
                        key=feature,
                        help=f"Range: {properties['min']}-{properties.get('max', 'N/A')} {properties['unit']}"
                    )
                input_values[feature] = value
    
    # 按 feature_order 顺序生成特征列表
    feature_values = [input_values[feature] for feature in feature_order]

with col2:
    st.header("Subphenotype Prediction")
    
    if st.button("🔍 Predict SA-AKI Subphenotype", type="primary", help="Click to calculate subphenotype probability"):
        try:
            # 转换为模型输入格式
            features = np.array([feature_values])
            
            # 进行预测 - 获取每个模型的正类概率
            prob1 = model1.predict_proba(features)[0][1]  # SP I
            prob2 = model2.predict_proba(features)[0][1]  # SP II
            prob3 = model3.predict_proba(features)[0][1]  # SP III
            prob4 = model4.predict_proba(features)[0][1]  # SP IV
            
            # 使用 softmax 归一化，使概率之和为 1
            probs_raw = np.array([prob1, prob2, prob3, prob4])
            probs_normalized = probs_raw / np.sum(probs_raw)
            
            probabilities = {
                "Subphenotype I": probs_normalized[0] * 100,
                "Subphenotype II": probs_normalized[1] * 100,
                "Subphenotype III": probs_normalized[2] * 100,
                "Subphenotype IV": probs_normalized[3] * 100
            }
            
            # 找出概率最高的亚型
            max_stage = max(probabilities, key=probabilities.get)
            max_prob = probabilities[max_stage]
            
            # ========== 显示主导预测结果 ==========
            st.markdown("### 🎯 Primary Prediction")
            
            # 根据最高概率和亚型设置颜色
            stage_color = sp_descriptions[max_stage]["color"]
            stage_prognosis = sp_descriptions[max_stage]["prognosis"]
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {stage_color}33 0%, {stage_color}66 100%); 
                        padding: 20px; border-radius: 10px; border-left: 6px solid {stage_color}; 
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;">
                <p style="font-size:14px; margin:0; color: #555; font-weight: bold;">PREDICTED SUBPHENOTYPE:</p>
                <h2 style="margin:8px 0; color: #333;">{max_stage}</h2>
                <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                    <div>
                        <p style="font-size:14px; margin:5px 0; color: #666;">Prognosis: <span style="font-weight: bold; color: {stage_color};">{stage_prognosis}</span></p>
                    </div>
                    <div style="text-align: right;">
                        <p style="font-size:13px; margin:5px 0; color: #888;">Prediction Confidence:</p>
                        <h3 style="margin:0; color: {stage_color};">{max_prob:.1f}%</h3>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # ========== 显示所有亚型的详细概率 ==========
            st.markdown("### 📊 Detailed Probability Distribution")
            
            # 创建概率表格
            prob_data = []
            for stage in ["Subphenotype I", "Subphenotype II", "Subphenotype III", "Subphenotype IV"]:
                prob_data.append({
                    "Subphenotype": stage,
                    "Probability": f"{probabilities[stage]:.2f}%",
                    "Prognosis": sp_descriptions[stage]["prognosis"]
                })
            
            df_prob = pd.DataFrame(prob_data)
            st.dataframe(df_prob, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # ========== 显示每个亚型的详细卡片 ==========
            st.markdown("### 📋 Subphenotype Characteristics")
            
            for stage in ["Subphenotype I", "Subphenotype II", "Subphenotype III", "Subphenotype IV"]:
                prob = probabilities[stage]
                color = sp_descriptions[stage]["color"]
                prognosis = sp_descriptions[stage]["prognosis"]
                
                # 创建卡片
                st.markdown(f"""
                <div class="stage-card" style="border-left-color: {color};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div>
                            <h4 style="margin: 0; color: #333;">{stage}</h4>
                        </div>
                        <div style="text-align: right;">
                            <span style="background-color: {color}; color: white; padding: 4px 12px; 
                                         border-radius: 20px; font-size: 12px; font-weight: bold;">{stage.split(' ')[1]}</span>
                        </div>
                    </div>
                    
                    <p style="margin: 10px 0 0 0; color: #666; font-size: 13px;">Prognosis: {prognosis}</p>
                    
                    <div style="margin-top: 15px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <div>
                                <span style="font-weight: bold; color: #333;">Prediction Probability</span>
                            </div>
                            <span style="font-weight: bold; color: {color}; font-size: 16px;">{prob:.2f}%</span>
                        </div>
                        <div class="probability-bar">
                            <div class="probability-fill" style="width: {prob}%; background-color: {color};">
                                {prob:.1f}%
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # ========== 显示概率分布图表 ==========
            st.markdown("### 📈 Probability Distribution Visualization")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # 柱状图
            stages = list(probabilities.keys())
            probs = list(probabilities.values())
            colors_list = [sp_descriptions[stage]["color"] for stage in stages]
            
            bars = ax1.bar(range(len(stages)), probs, color=colors_list, edgecolor='black', linewidth=1.5, alpha=0.8)
            ax1.set_ylabel('Probability (%)', fontsize=12, fontweight='bold')
            ax1.set_title('SA-AKI Subphenotype Probability Distribution', fontsize=13, fontweight='bold')
            ax1.set_xticks(range(len(stages)))
            ax1.set_xticklabels([f"{s.split(' ')[1]}" for s in stages], fontsize=11)
            ax1.set_ylim(0, max(probs) * 1.2)
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
            
            # 添加数值标签
            for i, (stage, prob) in enumerate(zip(stages, probs)):
                ax1.text(i, prob + 1, f'{prob:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            # 饼图
            ax2.pie(probs, labels=[f"{s}\n({prob:.1f}%)" for s, prob in zip(stages, probs)], 
                   colors=colors_list, autopct='', startangle=90, 
                   textprops={'fontsize': 10, 'fontweight': 'bold'},
                   wedgeprops={'edgecolor': 'black', 'linewidth': 1.5, 'alpha': 0.8})
            ax2.set_title('Probability Distribution (Pie Chart)', fontsize=13, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("---")
            
            # ========== 研究背景信息 ==========
            st.markdown("### 📚 Study Background")
            with st.expander("View Study Details"):
                st.markdown("""
                **Study Design:** Retrospective cohort study using MIMIC-IV v3.1 database (N=9,029 adult SA-AKI patients)
                
                **Methodology:**
                - Multi-algorithm consensus clustering (K-means, hierarchical clustering, K-medoids)
                - 19 clinical variables extracted within ±24 hours of SA-AKI diagnosis
                - Cluster stability assessed via consensus strength, entropy, silhouette analysis
                - SHAP analysis for feature interpretability
                
                **Key Findings:**
                - Four distinct SA-AKI subphenotypes with significantly different outcomes (log-rank P<0.0001)
                
                **Clinical Implications:**
                - Enables precision phenotyping for risk stratification
                - Supports targeted therapeutic trials
                - Identifies high-risk patients warranting early aggressive intervention
                """)
                
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.exception(e)
    else:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px dashed #ddd;">
            <h4 style="color: #666; margin-top: 0;">Ready for Prediction</h4>
            <p style="color: #888;">Enter patient clinical parameters in the left panel and click the prediction button above.</p>
            <p style="color: #888; font-size: 12px;">Note: All values should be measured within ±24 hours of SA-AKI diagnosis.</p>
        </div>
        """, unsafe_allow_html=True)

# 页脚
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px; line-height: 1.6;">
<p><strong>SA-AKI:</strong> Sepsis-Associated Acute Kidney Injury </p>
<p><strong>Reference:</strong> Chen S, Xu XC, Li G, et al. Multi-algorithm consensus clustering identifies four subphenotypes in sepsis-associated acute kidney injury. </p>
<p><strong>Database:</strong> MIMIC-IV v3.1 (Medical Information Mart for Intensive Care IV)</p>
<p><strong>⚠️ Disclaimer:</strong> This prediction tool is for clinical decision support and research purposes only. Clinical judgment should always supersede algorithmic predictions.</p>
<p><strong>For questions or technical support:</strong> Contact the corresponding author: Heng Li, M.D., Ph.D. (lh12818@163.com)</p>
</div>

""", unsafe_allow_html=True)
