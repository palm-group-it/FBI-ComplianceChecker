import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import io
from typing import Tuple

st.set_page_config(page_title="FBI Insurer-Mix Deviation Check", layout="wide")


def compute_outliers_count_only(df: pd.DataFrame, threshold: float, min_contracts: int = 0) -> pd.DataFrame:
    """
    Compute insurer mix deviations by count only.
    
    Args:
        df: DataFrame with columns 'UkKodja1', 'Megnevezés', 'RövidNév'
        threshold: Deviation threshold in percentage points
        min_contracts: Minimum number of contracts an agent must have in a line of business
                      to be included in the flagged results (baseline still uses all agents)
        
    Returns:
        DataFrame with flagged deviations
    """
    if min_contracts < 0:
        raise ValueError("min_contracts must be non-negative")
        
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
    
    if min_contracts > 0:
        results = results[results['agent_line_total'] >= min_contracts].copy()
    
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


def create_distribution_chart(agent_id: str, lob: str, source_df: pd.DataFrame) -> go.Figure:
    """
    Create a visualization comparing an agent's insurer distribution to baseline for a specific line of business.
    
    Args:
        agent_id: The agent ID to visualize
        lob: Line of business to analyze
        source_df: Original DataFrame with columns 'UkKodja1', 'Megnevezés', 'RövidNév'
        
    Returns:
        Plotly figure showing distribution comparison
    """
    required_cols = ['UkKodja1', 'Megnevezés', 'RövidNév']
    df = source_df[required_cols].copy()
    df.columns = ['agent_id', 'line_of_business', 'insurer']
    
    df['agent_id'] = df['agent_id'].fillna('(Missing)')
    df['line_of_business'] = df['line_of_business'].fillna('(Missing)')
    df['insurer'] = df['insurer'].fillna('(Missing)')
    
    lob_data = df[df['line_of_business'] == lob].copy()
    
    baseline = (
        lob_data.groupby('insurer', dropna=False)
        .size()
        .reset_index(name='base_count')
    )
    baseline['base_share'] = (baseline['base_count'] / baseline['base_count'].sum()) * 100
    
    agent_lob_data = lob_data[lob_data['agent_id'] == agent_id].copy()
    
    if len(agent_lob_data) == 0:
        return None
    
    agent_dist = (
        agent_lob_data.groupby('insurer', dropna=False)
        .size()
        .reset_index(name='agent_count')
    )
    agent_dist['agent_share'] = (agent_dist['agent_count'] / agent_dist['agent_count'].sum()) * 100
    
    comparison = baseline.merge(agent_dist, on='insurer', how='outer').fillna(0)
    comparison = comparison.sort_values('base_share', ascending=False)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Company Baseline',
        x=comparison['insurer'],
        y=comparison['base_share'],
        marker_color='lightblue',
        text=comparison['base_share'].round(1),
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Company: %{y:.1f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name=f'Agent {agent_id}',
        x=comparison['insurer'],
        y=comparison['agent_share'],
        marker_color='coral',
        text=comparison['agent_share'].round(1),
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Agent: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Insurer Distribution: Agent {agent_id} vs Company Baseline<br>Line of Business: {lob}',
        xaxis_title='Insurer',
        yaxis_title='Share %',
        barmode='group',
        height=500,
        hovermode='x unified'
    )
    
    return fig


def compute_sensitivity_analysis(df: pd.DataFrame, threshold_range: list, min_contracts: int = 0) -> pd.DataFrame:
    """
    Compute sensitivity analysis across multiple thresholds.
    
    Args:
        df: Source DataFrame
        threshold_range: List of threshold values to test
        min_contracts: Minimum contracts filter
        
    Returns:
        DataFrame with sensitivity analysis results
    """
    sensitivity_results = []
    
    for threshold in threshold_range:
        results = compute_outliers_count_only(df, threshold, min_contracts)
        
        sensitivity_results.append({
            'Threshold': threshold,
            'Total Deviations': len(results),
            'UP Deviations': (results['Direction'] == 'UP').sum() if len(results) > 0 else 0,
            'DOWN Deviations': (results['Direction'] == 'DOWN').sum() if len(results) > 0 else 0,
            'Unique Agents': results['Agent ID'].nunique() if len(results) > 0 else 0
        })
    
    return pd.DataFrame(sensitivity_results)


def create_sensitivity_chart(sensitivity_df: pd.DataFrame) -> go.Figure:
    """
    Create a line chart showing sensitivity analysis.
    
    Args:
        sensitivity_df: DataFrame with sensitivity analysis results
        
    Returns:
        Plotly figure showing sensitivity trends
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=sensitivity_df['Threshold'],
        y=sensitivity_df['Total Deviations'],
        mode='lines+markers',
        name='Total Deviations',
        line=dict(color='purple', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=sensitivity_df['Threshold'],
        y=sensitivity_df['UP Deviations'],
        mode='lines+markers',
        name='UP Deviations',
        line=dict(color='green', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=sensitivity_df['Threshold'],
        y=sensitivity_df['DOWN Deviations'],
        mode='lines+markers',
        name='DOWN Deviations',
        line=dict(color='red', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Sensitivity Analysis: Deviations vs Threshold',
        xaxis_title='Threshold (percentage points)',
        yaxis_title='Number of Deviations',
        height=500,
        hovermode='x unified'
    )
    
    return fig


def create_summary_dashboard(results: pd.DataFrame) -> dict:
    """
    Create summary statistics and top insights from flagged deviations.
    
    Args:
        results: DataFrame with flagged deviations
        
    Returns:
        Dictionary with summary statistics and top lists
    """
    summary = {}
    
    summary['top_agents'] = (
        results.groupby('Agent ID')['Difference (pp)']
        .apply(lambda x: x.abs().max())
        .nlargest(10)
        .reset_index()
    )
    summary['top_agents'].columns = ['Agent ID', 'Max Deviation (pp)']
    
    summary['affected_insurers'] = (
        results.groupby('Insurer')
        .size()
        .nlargest(10)
        .reset_index()
    )
    summary['affected_insurers'].columns = ['Insurer', 'Number of Deviations']
    
    summary['top_lobs'] = (
        results.groupby('Line of Business')
        .size()
        .nlargest(10)
        .reset_index()
    )
    summary['top_lobs'].columns = ['Line of Business', 'Number of Deviations']
    
    return summary


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
            
            analysis_mode = st.radio(
                "Analysis Mode",
                ["Single Threshold", "Multi-Threshold Sensitivity Analysis"],
                horizontal=True
            )
            
            if analysis_mode == "Single Threshold":
                col1, col2 = st.columns(2)
                with col1:
                    threshold = st.number_input(
                        "Deviation threshold (percentage points)",
                        min_value=0.0,
                        max_value=100.0,
                        value=10.0,
                        step=0.5,
                        help="App will flag deviations above +threshold or below −threshold"
                    )
                with col2:
                    min_contracts = st.number_input(
                        "Minimum contracts per agent (in line of business)",
                        min_value=0,
                        max_value=1000,
                        value=0,
                        step=1,
                        help="Filter out agents with fewer contracts to focus on statistically significant cases"
                    )
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    threshold_min = st.number_input(
                        "Min threshold (pp)",
                        min_value=0.0,
                        max_value=50.0,
                        value=5.0,
                        step=0.5
                    )
                with col2:
                    threshold_max = st.number_input(
                        "Max threshold (pp)",
                        min_value=0.0,
                        max_value=100.0,
                        value=25.0,
                        step=0.5
                    )
                with col3:
                    threshold_step = st.number_input(
                        "Step size (pp)",
                        min_value=0.5,
                        max_value=10.0,
                        value=2.5,
                        step=0.5
                    )
                
                min_contracts = st.number_input(
                    "Minimum contracts per agent (in line of business)",
                    min_value=0,
                    max_value=1000,
                    value=0,
                    step=1,
                    help="Filter out agents with fewer contracts to focus on statistically significant cases"
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
                        if analysis_mode == "Multi-Threshold Sensitivity Analysis":
                            if threshold_step <= 0:
                                st.error("Step size must be greater than 0.")
                            elif threshold_min > threshold_max:
                                st.error("Minimum threshold must be less than or equal to maximum threshold.")
                            elif threshold_step > (threshold_max - threshold_min):
                                st.error("Step size is too large for the given range.")
                            else:
                                threshold_range = []
                                current = threshold_min
                                while current <= threshold_max + 1e-9:
                                    threshold_range.append(round(current, 2))
                                    current += threshold_step
                                
                                if len(threshold_range) == 0:
                                    st.error("Invalid threshold range. Please check your min, max, and step values.")
                                else:
                                    st.info(f"Running sensitivity analysis for {len(threshold_range)} threshold values...")
                                    
                                    sensitivity_df = compute_sensitivity_analysis(df, threshold_range, min_contracts)
                                    
                                    st.subheader("📊 Sensitivity Analysis Results")
                                    
                                    col1, col2 = st.columns([2, 1])
                                    with col1:
                                        sensitivity_chart = create_sensitivity_chart(sensitivity_df)
                                        st.plotly_chart(sensitivity_chart, use_container_width=True)
                                    with col2:
                                        st.markdown("**Summary Table**")
                                        st.dataframe(sensitivity_df, hide_index=True, use_container_width=True)
                                    
                                    st.info("💡 Tip: Lower thresholds will flag more deviations, while higher thresholds focus on more extreme cases.")
                        
                        else:
                            results = compute_outliers_count_only(df, threshold, min_contracts)
                            
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
                                
                                st.subheader("📊 Summary Dashboard")
                                summary = create_summary_dashboard(results)
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.markdown("**Top Deviating Agents**")
                                    st.dataframe(summary['top_agents'], hide_index=True, use_container_width=True)
                                with col2:
                                    st.markdown("**Most Affected Insurers**")
                                    st.dataframe(summary['affected_insurers'], hide_index=True, use_container_width=True)
                                with col3:
                                    st.markdown("**Lines of Business with Most Deviations**")
                                    st.dataframe(summary['top_lobs'], hide_index=True, use_container_width=True)
                                
                                st.divider()
                                st.subheader("📈 Distribution Visualizations")
                                
                                with st.expander("View Agent Distribution Charts", expanded=False):
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        unique_agents = sorted(results['Agent ID'].unique())
                                        selected_agent = st.selectbox(
                                            "Select an agent to visualize",
                                            unique_agents,
                                            key="agent_viz_select"
                                        )
                                    with col_b:
                                        agent_lobs = sorted(results[results['Agent ID'] == selected_agent]['Line of Business'].unique())
                                        selected_lob = st.selectbox(
                                            "Select line of business",
                                            agent_lobs,
                                            key="lob_viz_select"
                                        )
                                    
                                    if selected_agent and selected_lob:
                                        fig = create_distribution_chart(selected_agent, selected_lob, df)
                                        if fig:
                                            st.plotly_chart(fig, use_container_width=True)
                                        else:
                                            st.info("No data available for this agent in this line of business.")
                                
                                st.divider()
                                st.subheader("📋 Detailed Flagged Deviations")
                                
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
