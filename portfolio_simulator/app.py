import streamlit as st
import pandas as pd
from data.data_handler import DataHandler
from portfolio.portfolio_manager import PortfolioManager
from execution.execution_handler import ExecutionHandler

st.set_page_config(layout="wide")

@st.cache_resource
def load_data_handler():
    """Loads the data handler, cached to prevent re-downloading."""
    print("Initializing DataHandler...")
    return DataHandler(
        assets_filepath='assets.txt',
        start_date='2004-01-01',
        end_date='2024-01-01',
        data_dir='data'
    )

def initialize_session_state():
    """Initializes the session state for the simulation."""
    if 'initialized' not in st.session_state:
        print("Initializing session state...")
        data_handler = load_data_handler()
        symbols = data_handler.symbols
        
        st.session_state.initialized = True
        st.session_state.data_handler = data_handler
        st.session_state.symbols = symbols
        st.session_state.portfolio_manager = PortfolioManager(symbols, initial_cash=1000000.0)
        st.session_state.execution_handler = ExecutionHandler()
        st.session_state.current_day_index = 0
        st.session_state.market_data = data_handler.get_data()
        st.session_state.log = []

# --- Main App ---
initialize_session_state()

# --- UI Layout ---
st.title("Interactive Portfolio Simulator")

# Sidebar for Portfolio Status
with st.sidebar:
    st.header("Portfolio Status")
    pm = st.session_state.portfolio_manager
    
    # Get the latest portfolio value
    if not pm.portfolio_value.empty:
        latest_value = pm.portfolio_value['value'].iloc[-1]
        st.metric("Total Value", f"${latest_value:,.2f}")
    else:
        st.metric("Total Value", f"${pm.initial_cash:,.2f}")

    st.metric("Cash Balance", f"${pm.cash_balance:,.2f}")

    st.subheader("Current Holdings")
    holdings_df = pd.DataFrame(list(pm.get_holdings().items()), columns=['Asset', 'Shares'])
    holdings_df = holdings_df[holdings_df['Shares'] > 0]
    st.dataframe(holdings_df, hide_index=True)
    
    st.subheader("Activity Log")
    st.text_area("Log", value="\n".join(st.session_state.log), height=200, disabled=True)


# Main content area
if st.session_state.current_day_index < len(st.session_state.market_data):
    
    # --- Date and Controls ---
    current_date = st.session_state.market_data.index[st.session_state.current_day_index]
    st.header(f"Date: {current_date.strftime('%Y-%m-%d')}")

    if st.button("Advance to Next Day"):
        st.session_state.current_day_index += 1
        st.rerun()

    # --- Market Data and Trading Form ---
    current_prices = st.session_state.market_data.iloc[st.session_state.current_day_index]
    
    st.subheader("Today's Market Prices")
    # Create a more robust DataFrame for displaying prices
    asset_price_data = []
    for symbol in st.session_state.symbols:
        price_series = current_prices.get(symbol)
        if price_series is not None and not price_series.empty:
            price = price_series.item()
            if pd.notna(price):
                asset_price_data.append({'Asset': symbol, 'Price': f"${price:.2f}"})
            else:
                asset_price_data.append({'Asset': symbol, 'Price': "N/A"})
        else:
            asset_price_data.append({'Asset': symbol, 'Price': "N/A"})
    
    prices_df = pd.DataFrame(asset_price_data)
    
    with st.form(key='trading_form'):
        st.dataframe(prices_df, hide_index=True, use_container_width=True)
        st.subheader("Place Your Orders")
        
        cols = st.columns(4)
        for i, symbol in enumerate(st.session_state.symbols):
            with cols[i % 4]:
                price_series = current_prices.get(symbol)
                if price_series is not None and not price_series.empty:
                    price = price_series.item()
                    if pd.notna(price):
                        st.number_input(f"Shares of {symbol}", key=f"trade_{symbol}", value=0, step=10)
                    else:
                        st.number_input(f"Shares of {symbol} (N/A)", key=f"trade_{symbol}", value=0, step=10, disabled=True)
                else:
                    st.number_input(f"Shares of {symbol} (N/A)", key=f"trade_{symbol}", value=0, step=10, disabled=True)

        submitted = st.form_submit_button("Execute Trades for Today")

        if submitted:
            trades_executed = False
            for symbol in st.session_state.symbols:
                quantity = st.session_state[f"trade_{symbol}"]
                if quantity != 0:
                    action = 'BUY' if quantity > 0 else 'SELL'
                    
                    # Check for sufficient shares to sell
                    if action == 'SELL' and st.session_state.portfolio_manager.holdings[symbol] < abs(quantity):
                        st.warning(f"Not enough shares of {symbol} to sell. You have {st.session_state.portfolio_manager.holdings[symbol]}.")
                        continue

                    signal = {'symbol': symbol, 'action': action, 'quantity': abs(quantity)}
                    transaction = st.session_state.execution_handler.execute_order(signal, current_prices)

                    if transaction:
                        # Check for sufficient cash to buy
                        if action == 'BUY' and st.session_state.portfolio_manager.cash_balance < (transaction['quantity'] * transaction['price']):
                            st.warning(f"Not enough cash to buy {transaction['quantity']} shares of {symbol}.")
                            continue
                        
                        st.session_state.portfolio_manager.update_holdings(
                            transaction['symbol'],
                            transaction['quantity'] if action == 'BUY' else -transaction['quantity'],
                            transaction['price']
                        )
                        log_msg = f"{current_date.date()}: {action} {transaction['quantity']} {symbol} @ ${transaction['price']:.2f}"
                        st.session_state.log.insert(0, log_msg)
                        trades_executed = True

            if trades_executed:
                st.success("Trades executed successfully!")
            else:
                st.info("No valid trades were executed.")
            
            # Always record portfolio value for the day, even if no trades
            st.session_state.portfolio_manager.record_portfolio_value(current_date, current_prices)
            st.rerun()

else:
    st.success("Simulation Finished!")
    st.balloons()

# --- Performance Chart ---
st.header("Portfolio Performance")
if not st.session_state.portfolio_manager.portfolio_value.empty:
    st.line_chart(st.session_state.portfolio_manager.portfolio_value)
else:
    st.info("No portfolio data yet. Start the simulation by advancing to the next day.")
