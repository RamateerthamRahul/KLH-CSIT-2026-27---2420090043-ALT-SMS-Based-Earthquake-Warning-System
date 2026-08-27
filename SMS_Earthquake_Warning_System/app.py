import os,joblib,pandas as pd,streamlit as st
st.set_page_config(page_title="SMS Earthquake Warning System",page_icon="📳",layout="wide")
st.title("📳 SMS-Based Earthquake Warning System")
st.caption("Educational prototype: ML warning classification + SMS-style alert simulation")
if not os.path.exists("models/earthquake_warning_model.joblib"):
    st.warning("Run: python train_model.py"); st.stop()
model=joblib.load("models/earthquake_warning_model.joblib"); df=pd.read_csv("data/earthquake_warning_data.csv")
st.sidebar.header("Earthquake Sensor Input")
vals={}
vals["pga_g"]=st.sidebar.number_input("PGA (g)",0.,3.,.35,.01)
vals["pga_growth"]=st.sidebar.number_input("PGA Growth",0.,2.,.25,.01)
vals["p_wave_sec"]=st.sidebar.number_input("P-Wave Arrival (sec)",0.,15.,3.,.1)
vals["depth_km"]=st.sidebar.number_input("Depth (km)",0.,700.,15.,.5)
vals["distance_km"]=st.sidebar.number_input("Distance to Epicenter (km)",0.,500.,30.,1.)
vals["stations"]=st.sidebar.number_input("Active Stations",1,500,20)
vals["noise_level"]=st.sidebar.number_input("Sensor Noise",0.,2.,.08,.01)
phone=st.sidebar.text_input("Recipient phone (demo only)","+91XXXXXXXXXX")
x=pd.DataFrame([vals])
if st.button("Analyze & Generate Warning",type="primary",use_container_width=True):
    pred=int(model.predict(x)[0]); prob=float(model.predict_proba(x)[0][1])
    a,b,c=st.columns(3); a.metric("Warning Probability",f"{prob*100:.1f}%"); b.metric("Decision","WARNING" if pred else "MONITOR"); c.metric("Recipient",phone)
    if pred:
        st.error("⚠️ EARTHQUAKE WARNING — Follow official emergency guidance.")
        st.code(f"EARTHQUAKE WARNING: Elevated seismic activity detected. Warning probability {prob*100:.1f}%. Move to a safe location and follow official instructions.")
        st.info("SMS status: SIMULATED / NOT SENT")
    else:
        st.success("🟢 MONITOR — No warning threshold reached in this educational model.")
        st.code("SMS status: No alert generated.")
st.divider(); a,b,c=st.columns(3); a.metric("Dataset Records",len(df)); b.metric("Features",7); c.metric("Target","Warning 0/1")
st.subheader("Dataset Preview"); st.dataframe(df.head(12),use_container_width=True)
sel=model.named_steps["feature_selection"]; st.subheader("Selected Features")
st.write(", ".join(df.drop(columns=["warning"]).columns[sel.get_support()]))
st.caption("Academic prototype only. Synthetic data and model are not suitable for real emergency warning decisions.")
