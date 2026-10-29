"""A completely serious guide to getting rich with this app. (It is not serious.)"""

import random

import streamlit as st

st.set_page_config(page_title="Get rich quick", page_icon="💸", layout="wide")

st.title("💸 How to Get Rich Using This App")
st.caption("A foolproof* guide. *Proof not included. Fools very much included.")

c1, c2, c3 = st.columns(3)
c1.metric("Guaranteed returns", "+∞%", "−100%")
c2.metric("Models that beat a random walk", "Rarely", "Sad noises", delta_color="inverse")
c3.metric("Author's net worth from this app", "$0.00", "Exactly as planned", delta_color="off")

st.header("The 7-step plan")
st.markdown(
    """
1. **Open the Machine learning tab.** Five models, 14 features, walk-forward validation. Very impressive.
2. **Notice that most of them lost to a random walk**, a model whose entire strategy is "predict nothing happens".
3. **Retrain them on a different date range** until one of them wins. Congratulations, you have invented *overfitting*,
   the leading cause of finance PhDs.
4. **Look at RSI.** If it's above 70, the stock is overbought, so sell. If it's below 30, it's oversold, so buy.
   If it's between 30 and 70, consult your horoscope.
5. **Go to Monte Carlo and run 5,000 simulations.** Ignore 4,999 of them. Screenshot the luckiest path.
   Set it as your lock screen. Manifest.
6. **Turn on "Allow short positions"** for that authentic "my losses are now theoretically unlimited" feeling.
7. **Set the trading cost slider to 0 bps**, because in your heart, fees are a social construct.
"""
)

st.header("Personalised risk assessment")
level = st.select_slider(
    "How confident are you in your next trade?",
    options=["Mildly curious", "Read one Reddit post", "Quit my job", "Remortgaged the house", "Sold the dog"],
    value="Read one Reddit post",
)
verdicts = {
    "Mildly curious": "Healthy. You may proceed to look at charts while drinking tea.",
    "Read one Reddit post": "You are now a 'market analyst'. Please update your LinkedIn.",
    "Quit my job": "Bold. Check the Max drawdown number on the front page before telling your parents.",
    "Remortgaged the house": "The Monte Carlo tab has a 5th percentile. Please go look at it. Slowly.",
    "Sold the dog": "The dog had a Sharpe ratio of infinity. Unconditional love, zero volatility. Buy it back.",
}
st.info(verdicts[level])

st.header("Hot stock tip generator")
st.caption("Powered by an advanced proprietary algorithm called random.choice().")
TIPS = [
    "Buy high, sell low. It's worked for millions of people.",
    "The MACD histogram crossed zero. That means something. Probably. To someone.",
    "Volatility is just the stock showing enthusiasm.",
    "If the price is above the 200-day moving average, it's going up. Until it isn't.",
    "Diversify: buy the same stock on two different brokerage apps.",
    "Past performance does not guarantee future results, but have you seen the past performance?",
    "The trend line says +40% a year forever. Lines never lie. Lines have never once lied.",
    "Correlation is 0.98, so if one goes down the other definitely won't. Wait.",
    "Sell everything on Friday, buy it all back on Monday. Call it a 'weekend hedge'.",
    "A 51% direction hit rate is basically a money printer. Ignore the trading costs. Ignore them.",
]
if st.button("🔮 Generate a guaranteed tip", type="primary"):
    st.session_state["tip"] = random.choice(TIPS)
    st.balloons()
if "tip" in st.session_state:
    st.success(st.session_state["tip"])

st.header("Frequently asked questions")
with st.expander("The Gradient boosting model had +2% skill. Should I invest my savings?"):
    st.write(
        "Run it again with a different date range and watch the 2% evaporate. That's the market's way "
        "of saying 'no' politely."
    )
with st.expander("The Monte Carlo says 63% chance of ending higher. Is that good?"):
    st.write(
        "It means that in a universe where the next three months behave exactly like the last two years, "
        "you'd usually be fine. We do not live in that universe. Nobody does."
    )
with st.expander("Why is the default forecast 'no change'?"):
    st.write("Because 'no change' kept beating the fancy models, and we respect a winner.")

st.divider()
st.subheader("🤫 The actual secret")
st.markdown(
    """
The boring truth that this app keeps proving: for liquid stocks, short-term prediction is extremely hard,
and costs eat most edges. What has historically worked for ordinary investors is unexciting:
**spend less than you earn, invest regularly in low-cost diversified funds, hold for years, and don't panic-sell.**

Compare SPY with any single stock over 10 years on the Compare tab and look at the volatility and max drawdown columns.
That's the whole lesson.
"""
)
st.caption("This page is a joke. Nothing in this app is financial advice. Please don't actually sell the dog. 🐕")
