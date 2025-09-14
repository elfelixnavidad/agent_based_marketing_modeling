import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add parent directory to path to access src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.environment.marketing_simulation import MarketingSimulation
from src.data_collection.metrics_collector import MetricsCollector


class MarketingDashboard:
    def __init__(self):
        self.simulation = None
        self.simulation_running = False
        
    def run(self):
        """Run the Streamlit dashboard"""
        st.set_page_config(
            page_title="Agent-Based Marketing Simulation",
            page_icon="📊",
            layout="wide"
        )
        
        st.title("📊 Agent-Based Marketing Simulation Dashboard")
        st.markdown("Interactive dashboard for mobile gaming UA strategy simulation")
        
        # Initialize session state
        if 'simulation' not in st.session_state:
            st.session_state.simulation = None
        if 'simulation_history' not in st.session_state:
            st.session_state.simulation_history = []
        
        # Sidebar for controls
        with st.sidebar:
            st.header("Simulation Controls")
            self._render_simulation_controls()
            
            if st.session_state.simulation:
                st.header("Budget Allocation")
                self._render_budget_controls()
                
                st.header("UA Channel Settings")
                self._render_channel_settings()
        
        # Main content area
        if st.session_state.simulation:
            self._render_dashboard()
        else:
            self._render_welcome_screen()
    
    def _render_simulation_controls(self):
        """Render simulation setup controls"""
        with st.expander("Simulation Setup", expanded=True):
            num_agents = st.slider("Number of Agents", 1000, 50000, 10000, 1000)
            initial_budget = st.number_input("Initial Budget ($)", 1000.0, 1000000.0, 100000.0, 1000.0)
            simulation_days = st.slider("Simulation Days", 30, 365, 90, 30)
            
            if st.button("Initialize Simulation", type="primary"):
                with st.spinner("Initializing simulation..."):
                    self.simulation = MarketingSimulation(
                        num_agents=num_agents,
                        initial_budget=initial_budget
                    )
                    st.session_state.simulation = self.simulation
                    st.session_state.simulation_history = []
                    st.success("Simulation initialized!")
        
        if st.session_state.simulation:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Run Simulation"):
                    self._run_simulation(simulation_days)
            with col2:
                if st.button("Step Forward"):
                    self._step_simulation()
    
    def _render_budget_controls(self):
        """Render budget allocation controls"""
        ua_manager = st.session_state.simulation.ua_manager
        current_budgets = ua_manager.get_channel_budgets()
        
        st.write("Current Budget Allocation:")
        total_budget = sum(current_budgets.values())
        
        for channel, budget in current_budgets.items():
            percentage = (budget / total_budget * 100) if total_budget > 0 else 0
            st.write(f"- {channel.replace('_', ' ').title()}: ${budget:,.0f} ({percentage:.1f}%)")
        
        st.write("Adjust Budgets:")
        new_budgets = {}
        remaining_budget = total_budget
        
        for channel in current_budgets.keys():
            if channel != 'owned_channels':  # Skip owned channels for budget allocation
                default_budget = current_budgets[channel]
                new_budget = st.number_input(
                    f"{channel.replace('_', ' ').title()} Budget ($)",
                    0.0, float(total_budget), float(default_budget), 1000.0,
                    key=f"budget_{channel}"
                )
                new_budgets[channel] = new_budget
        
        if st.button("Apply Budget Changes"):
            ua_manager.reallocate_budget(new_budgets)
            st.success("Budget allocation updated!")
    
    def _render_channel_settings(self):
        """Render UA channel configuration"""
        ua_manager = st.session_state.simulation.ua_manager
        
        for channel_name, channel in ua_manager.channels.items():
            with st.expander(f"{channel_name.replace('_', ' ').title()} Settings"):
                if hasattr(channel, 'cpi'):
                    current_cpi = max(float(channel.cpi), 0.1)  # Ensure minimum of 0.1
                    channel.cpi = st.number_input(
                        f"CPI ($)",
                        0.1, 10.0, current_cpi, 0.1,
                        key=f"cpi_{channel_name}"
                    )
                
                if hasattr(channel, 'ctr'):
                    current_ctr = max(float(channel.ctr), 0.0)  # Ensure minimum of 0.0
                    channel.ctr = st.slider(
                        f"Click-Through Rate",
                        0.0, 0.2, current_ctr, 0.005,
                        key=f"ctr_{channel_name}"
                    )
                
                if hasattr(channel, 'conversion_rate'):
                    current_conversion = max(float(channel.conversion_rate), 0.0)  # Ensure minimum of 0.0
                    channel.conversion_rate = st.slider(
                        f"Conversion Rate",
                        0.0, 0.3, current_conversion, 0.01,
                        key=f"conversion_{channel_name}"
                    )
    
    def _render_welcome_screen(self):
        """Render welcome screen for new users"""
        st.markdown("""
        ## Welcome to the Agent-Based Marketing Simulation!
        
        This dashboard allows you to simulate and analyze mobile gaming user acquisition strategies using agent-based modeling.
        
        ### Key Features:
        - **Agent-Based Modeling**: Simulate individual user behaviors and decision-making
        - **Multiple UA Channels**: Model paid social, video ads, search ads, and owned channels
        - **Real-time Analytics**: Monitor KPIs, retention curves, and channel performance
        - **Budget Optimization**: Test different budget allocation scenarios
        - **Cohort Analysis**: Analyze user retention and LTV patterns
        
        ### Getting Started:
        1. Use the sidebar to configure your simulation parameters
        2. Set initial budget and number of agents
        3. Click "Initialize Simulation" to create your simulation
        4. Adjust budget allocation and channel settings
        5. Run the simulation to see results
        
        ### Business Questions You Can Answer:
        - What's the optimal budget split between different UA channels?
        - How does diminishing returns affect campaign performance?
        - What's the impact of organic growth (k-factor) on paid campaigns?
        - How do different retention strategies affect LTV?
        """)
    
    def _render_dashboard(self):
        """Render the main dashboard"""
        # Get simulation data
        collector = st.session_state.simulation.data_collector
        kpis = collector.get_kpi_summary()
        
        # Key metrics row
        st.header("Key Performance Indicators")
        self._render_kpi_cards(kpis)
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Time Series", 
            "🎯 Conversion Funnel", 
            "📊 Channel Performance", 
            "👥 Cohort Analysis", 
            "🔮 What-If Scenarios"
        ])
        
        with tab1:
            self._render_time_series_view()
        
        with tab2:
            self._render_funnel_view()
        
        with tab3:
            self._render_channel_view()
        
        with tab4:
            self._render_cohort_view()
        
        with tab5:
            self._render_scenario_view()
    
    def _render_kpi_cards(self, kpis):
        """Render KPI cards"""
        if not kpis:
            st.warning("Run simulation to see KPIs")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Revenue", f"${kpis['total_revenue']:,.0f}")
            st.metric("LTV", f"${kpis['ltv']:.2f}")
        
        with col2:
            st.metric("Total Installs", f"{kpis['total_installs']:,}")
            st.metric("CAC", f"${kpis['cac']:.2f}")
        
        with col3:
            st.metric("MAU", f"{kpis['current_mau']:,}")
            st.metric("LTV:CAC Ratio", f"{kpis['ltv_cac_ratio']:.2f}")
        
        with col4:
            st.metric("ARPU", f"${kpis['arpu']:.2f}")
            st.metric("Day 7 Retention", f"{kpis['day7_retention']:.1%}")
    
    def _render_time_series_view(self):
        """Render time series charts"""
        st.subheader("Time Series Analysis")
        
        if st.session_state.simulation.data_collector.daily_metrics.empty:
            st.warning("No data available. Run simulation first.")
            return
        
        df = st.session_state.simulation.data_collector.daily_metrics
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('User Acquisition', 'Revenue Metrics', 'Retention & Activity', 'Budget & ROAS'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}]]
        )
        
        # User Acquisition
        fig.add_trace(
            go.Scatter(x=df['step'], y=df['installs_today'], name='Daily Installs', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['step'], y=df['cumulative_revenue'], name='Cumulative Revenue', line=dict(color='green')),
            row=1, col=1, secondary_y=True
        )
        
        # Revenue Metrics
        fig.add_trace(
            go.Scatter(x=df['step'], y=df['revenue_today'], name='Daily Revenue', line=dict(color='orange')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=df['step'], y=df['arpu'], name='ARPU', line=dict(color='purple')),
            row=1, col=2, secondary_y=True
        )
        
        # Retention & Activity
        fig.add_trace(
            go.Scatter(x=df['step'], y=df['installed_agents'] + df['paying_users'], name='Active Users', line=dict(color='red')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['step'], y=df['churns_today'], name='Daily Churns', line=dict(color='brown')),
            row=2, col=1, secondary_y=True
        )
        
        # Budget & ROAS
        fig.add_trace(
            go.Scatter(x=df['step'], y=df['daily_budget_spend'], name='Daily Spend', line=dict(color='gray')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=df['step'], y=df['roas'], name='ROAS', line=dict(color='pink')),
            row=2, col=2, secondary_y=True
        )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_funnel_view(self):
        """Render conversion funnel analysis"""
        st.subheader("Conversion Funnel")
        
        funnel_data = st.session_state.simulation.data_collector.get_funnel_analysis()
        
        # Create funnel chart
        funnel_stages = ['unaware', 'aware', 'installed', 'paying_user']
        funnel_values = [funnel_data[stage]['count'] for stage in funnel_stages]
        
        fig = go.Figure(go.Funnel(
            y=[stage.replace('_', ' ').title() for stage in funnel_stages],
            x=funnel_values,
            textinfo="value+percent initial"
        ))
        
        fig.update_layout(title="User Conversion Funnel")
        st.plotly_chart(fig, use_container_width=True)
        
        # Conversion rates
        st.subheader("Conversion Rates")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Aware → Install Rate", f"{funnel_data.get('aware_to_install_rate', 0):.1f}%")
        
        with col2:
            st.metric("Install → Paying Rate", f"{funnel_data.get('install_to_paying_rate', 0):.1f}%")
    
    def _render_channel_view(self):
        """Render channel performance analysis"""
        st.subheader("UA Channel Performance")
        
        channel_performance = st.session_state.simulation.ua_manager.get_performance_metrics()
        
        if not channel_performance:
            st.warning("No channel performance data available.")
            return
        
        # Create metrics dataframe
        metrics_data = []
        for channel, metrics in channel_performance.items():
            metrics_data.append({
                'Channel': channel.replace('_', ' ').title(),
                'Spend': metrics['spend'],
                'Installs': metrics['installs'],
                'CPI': metrics['cpi'],
                'CTR': metrics['ctr'],
                'ROAS': metrics['roas'],
            })
        
        df = pd.DataFrame(metrics_data)
        
        # Performance comparison chart
        fig = px.bar(df, x='Channel', y=['Installs', 'Spend'], 
                     title="Channel Performance: Installs vs Spend",
                     barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        # Efficiency metrics
        col1, col2 = st.columns(2)
        
        with col1:
            fig_cpi = px.bar(df, x='Channel', y='CPI', title="Cost Per Install by Channel")
            st.plotly_chart(fig_cpi, use_container_width=True)
        
        with col2:
            fig_roas = px.bar(df, x='Channel', y='ROAS', title="ROAS by Channel")
            st.plotly_chart(fig_roas, use_container_width=True)
        
        # Detailed metrics table
        st.subheader("Detailed Channel Metrics")
        st.dataframe(df.style.format({
            'Spend': '${:,.0f}',
            'Installs': '{:,}',
            'CPI': '${:.2f}',
            'CTR': '{:.2%}',
            'ROAS': '{:.2f}'
        }))
    
    def _render_cohort_view(self):
        """Render cohort analysis"""
        st.subheader("Cohort Analysis")
        
        cohort_data = st.session_state.simulation.data_collector.get_cohort_analysis()
        
        if not cohort_data:
            st.warning("No cohort data available. Run simulation for longer period.")
            return
        
        # Retention curves
        fig = go.Figure()
        
        for cohort_week, retention_data in cohort_data.items():
            days = [point['day'] for point in retention_data]
            rates = [point['retention_rate'] for point in retention_data]
            
            fig.add_trace(go.Scatter(
                x=days, y=rates,
                name=f'Week {cohort_week}',
                mode='lines+markers'
            ))
        
        fig.update_layout(
            title="Retention Curves by Cohort",
            xaxis_title="Days Since Install",
            yaxis_title="Retention Rate",
            yaxis_tickformat=".0%"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_scenario_view(self):
        """Render what-if scenario analysis"""
        st.subheader("What-If Scenario Analysis")
        
        st.markdown("""
        Test different budget allocation scenarios to see their potential impact.
        """)
        
        # Scenario selection
        scenario_type = st.selectbox(
            "Select Scenario Type",
            ["Budget Reallocation", "Channel Performance Changes", "Market Conditions"]
        )
        
        if scenario_type == "Budget Reallocation":
            self._render_budget_scenario()
        elif scenario_type == "Channel Performance Changes":
            self._render_performance_scenario()
        else:
            self._render_market_scenario()
    
    def _render_budget_scenario(self):
        """Render budget reallocation scenario"""
        st.write("Test different budget allocation strategies:")
        
        strategy = st.selectbox(
            "Budget Strategy",
            ["Equal Allocation", "Performance-Based", "High-ROAS Focus", "Growth Focus"]
        )
        
        if st.button("Run Scenario Analysis"):
            # Get current total budget
            ua_manager = st.session_state.simulation.ua_manager
            current_budgets = ua_manager.get_channel_budgets()
            total_budget = sum(current_budgets.values())
            
            # Calculate new allocation based on strategy
            if strategy == "Equal Allocation":
                paid_channels = [c for c in current_budgets.keys() if c != 'owned_channels']
                per_channel = total_budget // len(paid_channels)
                new_budgets = {channel: per_channel for channel in paid_channels}
            elif strategy == "Performance-Based":
                new_budgets = ua_manager.optimize_budget_allocation('roas')
            elif strategy == "High-ROAS Focus":
                new_budgets = ua_manager.optimize_budget_allocation('roas')
                # Double down on best performer
                if new_budgets:
                    best_channel = max(new_budgets, key=new_budgets.get)
                    new_budgets[best_channel] *= 1.5
            else:  # Growth Focus
                new_budgets = {
                    'paid_social': total_budget * 0.5,
                    'video_ads': total_budget * 0.3,
                    'search_ads': total_budget * 0.2
                }
            
            # Display recommended allocation
            st.subheader("Recommended Budget Allocation")
            for channel, budget in new_budgets.items():
                percentage = (budget / total_budget * 100) if total_budget > 0 else 0
                st.write(f"- {channel.replace('_', ' ').title()}: ${budget:,.0f} ({percentage:.1f}%)")
            
            if st.button("Apply This Allocation"):
                ua_manager.reallocate_budget(new_budgets)
                st.success("Budget allocation applied!")
    
    def _render_performance_scenario(self):
        """Render channel performance scenario"""
        st.write("Adjust channel performance parameters:")
        
        channel = st.selectbox(
            "Select Channel",
            ["paid_social", "video_ads", "search_ads"]
        )
        
        ua_manager = st.session_state.simulation.ua_manager
        selected_channel = ua_manager.channels.get(channel)
        
        if selected_channel:
            performance_change = st.slider(
                "Performance Multiplier",
                0.5, 2.0, 1.0, 0.1
            )
            
            if st.button("Apply Performance Change"):
                selected_channel.effectiveness_multiplier = performance_change
                st.success(f"Updated {channel} performance multiplier to {performance_change}x")
    
    def _render_market_scenario(self):
        """Render market conditions scenario"""
        st.write("Adjust market conditions:")
        
        market_factor = st.slider(
            "Market Conditions Multiplier",
            0.5, 2.0, 1.0, 0.1,
            help="1.0 = Normal conditions, <1.0 = Recession, >1.0 = Growth"
        )
        
        if st.button("Apply Market Conditions"):
            # Apply market factor to all channels
            ua_manager = st.session_state.simulation.ua_manager
            for channel in ua_manager.channels.values():
                if hasattr(channel, 'effectiveness_multiplier'):
                    base_multiplier = channel.config.get('effectiveness_multiplier', 1.0)
                    channel.effectiveness_multiplier = base_multiplier * market_factor
            
            st.success(f"Applied market conditions multiplier: {market_factor}x")
    
    def _run_simulation(self, days):
        """Run simulation for specified number of days"""
        if not st.session_state.simulation:
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(days):
            st.session_state.simulation.step()
            
            # Update progress
            progress = (i + 1) / days
            progress_bar.progress(progress)
            status_text.text(f"Running simulation... Day {i + 1}/{days}")
            
            # Update display every 10 days
            if (i + 1) % 10 == 0:
                st.rerun()
        
        progress_bar.progress(1.0)
        status_text.text("Simulation complete!")
        st.success(f"Simulation completed for {days} days")
        
        # Force refresh to show new data
        st.rerun()
    
    def _step_simulation(self):
        """Step simulation forward by one day"""
        if st.session_state.simulation:
            st.session_state.simulation.step()
            st.success("Stepped forward by 1 day")
            st.rerun()


if __name__ == "__main__":
    dashboard = MarketingDashboard()
    dashboard.run()