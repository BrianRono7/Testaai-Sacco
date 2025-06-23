import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.set_page_config(layout="wide")
    
    # Sidebar (for info only)
    st.sidebar.title("💼 Money Manager")
    st.sidebar.markdown("""
Welcome to your smart personal finance assistant!

**How it works:**
- Upload a transaction CSV file (must include 'Note', and optionally 'Date' and 'KES')
- See automated classification of Income vs Expense
- View charts and download results
    """)
    st.sidebar.markdown("---")
    st.sidebar.info("Need help? 📧 info@nerdwaretechnologies.com")

    st.title("Testaai Sacco Smart Transaction Classifier & Dashboard")

    # Load the classifier model
    try:
        model = joblib.load("income_expense_classifier.pkl")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return

    # MAIN upload section (not in sidebar)
    uploaded_file = st.file_uploader("Upload your transaction CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")

            if 'Note' not in df.columns:
                st.error("❌ Your file must contain a 'Note' column.")
                return

            # Predict income vs expense
            df['Predicted Type'] = model.predict(df['Note'])

            # Optional: Convert Date
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df['Month'] = df['Date'].dt.to_period('M')

            # Show predictions
            st.subheader("📄 Predictions Preview")
            preview_cols = ['Note', 'Predicted Type']
            if 'KES' in df.columns:
                preview_cols.append('KES')
            if 'Date' in df.columns:
                preview_cols.append('Date')
            st.dataframe(df[preview_cols])

            # Download predictions
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Results as CSV", data=csv, file_name="predictions.csv", mime="text/csv")

            # Summary KPIs
            if 'KES' in df.columns:
                income_total = df[df['Predicted Type'] == 'Income']['KES'].sum()
                expense_total = df[df['Predicted Type'] == 'Expense']['KES'].sum()

                st.subheader("📌 Summary")
                col1, col2 = st.columns(2)
                col1.metric("Total Income", f"KES {income_total:,.0f}")
                col2.metric("Total Expenses", f"KES {expense_total:,.0f}")

            # Visualizations
            st.subheader("📊 Visual Insights")

            # Pie Chart
            st.markdown("### Income vs Expense Breakdown")
            pie_data = df['Predicted Type'].value_counts()
            fig1, ax1 = plt.subplots()
            ax1.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
            ax1.axis('equal')
            st.pyplot(fig1)

            # Bar Chart of Amounts
            if 'KES' in df.columns:
                st.markdown("### Total Amount by Type")
                bar_data = df.groupby('Predicted Type')['KES'].sum().reset_index()
                fig2, ax2 = plt.subplots()
                sns.barplot(data=bar_data, x='Predicted Type', y='KES', ax=ax2)
                st.pyplot(fig2)

            # Monthly Volume Trends
            if 'Date' in df.columns:
                st.markdown("### Monthly Transaction Volume")
                time_data = df.groupby(['Month', 'Predicted Type']).size().unstack().fillna(0)
                st.line_chart(time_data)

        except Exception as e:
            st.error(f"❌ Failed to process file: {e}")
    else:
        st.info("📂 Upload a transaction CSV file to get started.")

if __name__ == "__main__":
    main()
