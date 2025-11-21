import streamlit as st
import pandas as pd
import io
from typing import Tuple

st.set_page_config(page_title="FBI Insurer-Mix Deviation Check", layout="wide")


def compute_outliers_count_only(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Compute insurer mix deviations by count only.
    
    Args:
        df: DataFrame with columns 'UkKodja1', 'Megnevezés', 'RövidNév'
        threshold: Deviation threshold in percentage points
        
    Returns:
        DataFrame with flagged deviations
    """
    required_cols = ['UkKodja1', 'Megnevezés', 'RövidNév']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    df = df[required_cols].copy()
    df.columns = ['agent_id', 'line_of_business', 'insurer']
    
    df['agent_id'] = df['agent_id'].fillna('(Missing)')
    df['line_of_business'] = df['line_of_business'].fillna('(Missing)')
    df['insurer'] = df['insurer'].fillna('(Missing)')
    
    baseline = (
        df.groupby(['line_of_business', 'insurer'], dropna=False)
        .size()
        .reset_index(name='base_count')
    )
    
    line_totals = (
        df.groupby('line_of_business', dropna=False)
        .size()
        .reset_index(name='line_total_count')
    )
    
    baseline = baseline.merge(line_totals, on='line_of_business')
    baseline['base_share'] = baseline['base_count'] / baseline['line_total_count']
    
    agent_counts = (
        df.groupby(['agent_id', 'line_of_business', 'insurer'], dropna=False)
        .size()
        .reset_index(name='agent_count')
    )
    
    agent_line_totals = (
        df.groupby(['agent_id', 'line_of_business'], dropna=False)
        .size()
        .reset_index(name='agent_line_total')
    )
    
    agent_data = agent_counts.merge(
        agent_line_totals, 
        on=['agent_id', 'line_of_business']
    )
    agent_data['agent_share'] = agent_data['agent_count'] / agent_data['agent_line_total']
    
    results = agent_data.merge(
        baseline,
        on=['line_of_business', 'insurer'],
        how='left'
    )
    
    results['diff_pp'] = (results['agent_share'] - results['base_share']) * 100
    
    results = results.dropna(subset=['diff_pp'])
    
    flagged = results[results['diff_pp'].abs() > threshold].copy()
    
    flagged['direction'] = flagged['diff_pp'].apply(lambda x: 'UP' if x > 0 else 'DOWN')
    
    flagged['company_share_pct'] = flagged['base_share'] * 100
    flagged['agent_share_pct'] = flagged['agent_share'] * 100
    
    output = flagged[[
        'agent_id', 'line_of_business', 'insurer',
        'base_count', 'line_total_count', 'company_share_pct',
        'agent_count', 'agent_line_total', 'agent_share_pct',
        'diff_pp', 'direction'
    ]].copy()
    
    output.columns = [
        'Agent ID', 'Line of Business', 'Insurer',
        'Company Count', 'Company Line Total', 'Company Share %',
        'Agent Count', 'Agent Line Total', 'Agent Share %',
        'Difference (pp)', 'Direction'
    ]
    
    output['abs_diff'] = output['Difference (pp)'].abs()
    output = output.sort_values(
        ['Agent ID', 'Line of Business', 'abs_diff'],
        ascending=[True, True, False]
    )
    output = output.drop(columns=['abs_diff'])
    
    return output


def style_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply color styling to the results DataFrame."""
    def highlight_diff(row):
        styles = [''] * len(row)
        diff_idx = df.columns.get_loc('Difference (pp)')
        direction_idx = df.columns.get_loc('Direction')
        
        if row['Direction'] == 'UP':
            styles[diff_idx] = 'background-color: #90EE90'
            styles[direction_idx] = 'background-color: #90EE90'
        elif row['Direction'] == 'DOWN':
            styles[diff_idx] = 'background-color: #FFB6C6'
            styles[direction_idx] = 'background-color: #FFB6C6'
        
        return styles
    
    return df.style.apply(highlight_diff, axis=1)


def main():
    st.title("🔍 FBI Insurer-Mix Deviation Check (Count-Only)")
    
    st.markdown("""
    Upload an Excel file to analyze which sales agents deviate significantly from the company's 
    overall insurer mix distribution within each line of business.
    """)
    
    uploaded_file = st.file_uploader(
        "Upload Excel file (.xlsx)", 
        type=['xlsx'],
        help="Select the Excel file containing contract data"
    )
    
    if uploaded_file is not None:
        try:
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            default_idx = 0
            if "DosszieAdatok282 - eredeti" in sheet_names:
                default_idx = sheet_names.index("DosszieAdatok282 - eredeti")
            
            selected_sheet = st.selectbox(
                "Select sheet to analyze",
                sheet_names,
                index=default_idx
            )
            
            threshold = st.number_input(
                "Deviation threshold (percentage points)",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=0.5,
                help="App will flag deviations above +threshold or below −threshold"
            )
            
            if st.button("🚀 Run Analysis", type="primary"):
                with st.spinner("Analyzing data..."):
                    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                    
                    st.info(f"Loaded {len(df):,} rows from sheet '{selected_sheet}'")
                    
                    required_cols = ['UkKodja1', 'Megnevezés', 'RövidNév']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    
                    if missing_cols:
                        st.error(f"Missing required columns: {', '.join(missing_cols)}")
                        st.write("Available columns:", list(df.columns))
                    else:
                        results = compute_outliers_count_only(df, threshold)
                        
                        if len(results) == 0:
                            st.warning(f"No deviations found above the threshold of ±{threshold} percentage points.")
                        else:
                            st.success(f"Found {len(results):,} flagged deviations")
                            
                            up_count = (results['Direction'] == 'UP').sum()
                            down_count = (results['Direction'] == 'DOWN').sum()
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Deviations", len(results))
                            with col2:
                                st.metric("UP Deviations", up_count)
                            with col3:
                                st.metric("DOWN Deviations", down_count)
                            
                            st.subheader("Flagged Deviations")
                            
                            styled_df = style_dataframe(results)
                            st.dataframe(
                                styled_df,
                                use_container_width=True,
                                height=600
                            )
                            
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                results.to_excel(writer, index=False, sheet_name='Deviations')
                                
                                workbook = writer.book
                                worksheet = writer.sheets['Deviations']
                                
                                up_format = workbook.add_format({'bg_color': '#90EE90'})
                                down_format = workbook.add_format({'bg_color': '#FFB6C6'})
                                
                                diff_col_idx = results.columns.get_loc('Difference (pp)')
                                direction_col_idx = results.columns.get_loc('Direction')
                                
                                for row_idx, direction in enumerate(results['Direction'], start=1):
                                    if direction == 'UP':
                                        worksheet.write(row_idx, diff_col_idx, 
                                                      results.iloc[row_idx - 1]['Difference (pp)'], 
                                                      up_format)
                                        worksheet.write(row_idx, direction_col_idx, 'UP', up_format)
                                    elif direction == 'DOWN':
                                        worksheet.write(row_idx, diff_col_idx, 
                                                      results.iloc[row_idx - 1]['Difference (pp)'], 
                                                      down_format)
                                        worksheet.write(row_idx, direction_col_idx, 'DOWN', down_format)
                            
                            excel_data = output.getvalue()
                            
                            st.download_button(
                                label="📥 Download Results as Excel",
                                data=excel_data,
                                file_name=f"fbi_outliers_threshold_{threshold}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)
    else:
        st.info("👆 Please upload an Excel file to begin analysis")


if __name__ == "__main__":
    main()
