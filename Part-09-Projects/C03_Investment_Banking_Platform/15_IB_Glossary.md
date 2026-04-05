# Module 15: Investment Banking Glossary

## Module Overview

This glossary defines **80+ investment banking and trading terms** for C++ developers
who don't come from finance. Every term includes a plain-language definition, the module
where it's used in this project, and — where relevant — a note on how the concept
maps to code.

**How to use this glossary:**
- When reading any module and you encounter a finance term, look it up here
- Each entry includes `(Module X)` cross-references showing where the term appears
- Terms are grouped by category, then alphabetical within each category
- Code-relevant notes explain how the financial concept translates to data structures

---

## Trading Terms

**Ask** — The lowest price at which a seller is willing to sell a security. Also called
the "offer." In the order book, asks are sorted ascending (lowest first). The best ask
is the top of the sell side. (Module 2: Order Book)

**Algo Trading** — Automated trading using computer programs that follow a defined set
of rules (algorithms) to place trades. Our entire platform is an algo trading system.
Algorithms range from simple (VWAP execution) to complex (statistical arbitrage).
(Module 12: System Integration)

**Bid** — The highest price at which a buyer is willing to buy a security. In the order
book, bids are sorted descending (highest first). The best bid is the top of the buy
side. The bid-ask spread is the difference between the best ask and best bid.
(Module 2: Order Book)

**Circuit Breaker** — An automatic halt in trading triggered when prices move too rapidly.
Exchanges implement circuit breakers to prevent flash crashes. In our platform, risk
limits serve a similar role — halting trading when losses exceed thresholds.
(Module 5: Risk Engine, Module 16: Regulatory)

**Clearing** — The process of reconciling trades between buyer and seller after execution.
A clearinghouse (like DTCC or LCH) guarantees settlement, reducing counterparty risk.
Our platform generates clearing-ready trade reports. (Module 11: Persistence)

**Dark Pool** — A private exchange where institutional investors trade large blocks
without revealing their order size to the public market. Dark pools reduce market impact
but raise transparency concerns. (Module 6: Execution Gateway)

**Day Order** — An order that expires at the end of the trading day if not filled.
The most common time-in-force setting. Contrast with GTC (Good Till Cancel).
(Module 2: Order Book)

**Fill** — A completed execution of an order. A fill includes the price, quantity, and
timestamp. An order can be partially filled (100 of 500 shares executed) or fully
filled. In code, fills update the Position Tracker. (Module 7: Position Tracker)

**FOK (Fill or Kill)** — An order that must be filled in its entirety immediately or
cancelled entirely. No partial fills allowed. Used for large institutional orders where
partial execution would create risk. (Module 2: Order Book)

**GTC (Good Till Cancel)** — An order that remains active until explicitly cancelled by
the trader or filled. Unlike day orders, GTC orders persist across trading sessions.
(Module 2: Order Book)

**Iceberg Order** — An order where only a small "visible" portion is displayed to the
market, while the larger "hidden" portion is automatically replenished as fills occur.
Used to minimize market impact when trading large quantities. (Module 2: Order Book)

**IOC (Immediate or Cancel)** — An order that must be filled immediately (partially or
fully); any unfilled portion is cancelled. Useful when you want immediate execution
without leaving a resting order. (Module 2: Order Book)

**Leverage** — Using borrowed capital to increase position size beyond what your own
capital allows. A 10× leverage means a $100K account controls $1M in assets. Amplifies
both gains and losses. (Module 5: Risk Engine)

**Liquidity** — The ability to buy or sell an asset without significantly moving its price.
High liquidity = tight spreads, many orders. Low liquidity = wide spreads, slippage risk.
In code, order book depth is a proxy for liquidity. (Module 2: Order Book)

**Lot Size** — The minimum quantity increment for an order. For US equities, typically
1 share. For FX, standard lot = 100,000 units. For futures, defined per contract spec.
Orders with quantities not divisible by lot size are rejected. (Module 2: Order Book)

**Margin** — The collateral deposited with a broker to cover potential losses. Initial
margin is required to open a position; maintenance margin is the minimum to keep it open.
A margin call occurs when account equity falls below maintenance. (Module 5: Risk Engine)

**Market Maker** — A firm that provides liquidity by continuously quoting bid and ask
prices, profiting from the spread. Market makers are required to maintain quotes within
defined spread limits. In code, a market maker strategy generates two-sided quotes.
(Module 3: Instruments, Module 12: System Integration)

**Notional** — The total face value of a position. For 100 shares at $150, notional =
$15,000. For a $1M bond at 98% of par, notional = $1M (market value = $980K). Risk
limits are often expressed in notional terms. (Module 5: Risk Engine)

**Settlement** — The actual exchange of securities for cash after a trade. US equities
settle T+1 (one business day after trade date). Some markets still use T+2. Until
settlement, the trade is "pending." (Module 11: Persistence)

**Short Selling** — Selling a security you don't own (borrowing it first), with the
obligation to buy it back later. Profit if the price falls; unlimited potential loss if
it rises. Short positions are represented as negative quantities in the Position Tracker.
(Module 7: Position Tracker)

**Slippage** — The difference between the expected fill price and the actual fill price.
Occurs in fast-moving markets or with large orders that consume multiple price levels.
Slippage = actual_price - expected_price. (Module 6: Execution Gateway)

**Spread** — The difference between the best ask and best bid prices. Tight spread =
liquid market. Wide spread = illiquid or volatile market. Spread = best_ask - best_bid.
A negative spread ("crossed book") should trigger immediate matching.
(Module 2: Order Book)

**Stop Loss** — An order placed to sell a position when the price drops to a specified
level, limiting losses. A stop at $95 on a position bought at $100 limits loss to ~5%.
Converts to a market order when triggered. (Module 2: Order Book)

**Take Profit** — An order placed to sell a position when the price reaches a target
profit level. A take-profit at $110 on a $100 position locks in ~10% gain. Paired with
stop-loss for risk management. (Module 2: Order Book)

**Tick Size** — The minimum price increment for a security. US equities: $0.01. ES
futures: $0.25. Prices must be multiples of tick size; non-conforming orders are rejected.
In code: `price = std::round(price / tick_size) * tick_size`. (Module 2: Order Book)

**TWAP (Time-Weighted Average Price)** — An execution algorithm that spreads a large
order evenly over a time period. If you want to buy 10,000 shares over 1 hour, TWAP
buys ~167 shares every minute. Minimizes market impact for time-insensitive orders.
(Module 12: System Integration)

**VWAP (Volume-Weighted Average Price)** — The average price weighted by volume traded
at each price level throughout the day. Used as a benchmark: if your execution VWAP is
better than the market VWAP, you got a good fill. `VWAP = Σ(price × volume) / Σ(volume)`.
(Module 7: Position Tracker, Module 12: System Integration)

---

## Instrument Terms

**Bond** — A fixed-income security where the issuer (government or corporation) borrows
money from the investor and pays periodic interest (coupons) plus the principal at
maturity. Bonds are priced as a percentage of par value (e.g., 98.5 = 98.5% of face).
(Module 3: Instruments)

**CDS (Credit Default Swap)** — A derivative contract where the buyer pays periodic
premiums to the seller, who agrees to compensate the buyer if a specified credit event
(default) occurs on a reference entity. Essentially insurance on bond default.
(Module 3: Instruments)

**Coupon** — The periodic interest payment on a bond, expressed as an annual percentage
of par value. A 5% coupon on a $1,000 bond pays $50/year (or $25 semi-annually).
`coupon_payment = par_value * coupon_rate / payments_per_year`. (Module 3: Instruments)

**Equity** — An ownership stake in a company, typically in the form of shares of stock.
Equities trade on exchanges (NYSE, NASDAQ) and have variable prices determined by supply
and demand. The simplest instrument type in our system. (Module 3: Instruments)

**Expiry** — The date on which an options or futures contract ceases to exist. After
expiry, the contract is either exercised (if in-the-money) or expires worthless. Time
decay (theta) accelerates as expiry approaches. (Module 3: Instruments, Module 4: Pricing)

**Forward** — A customized, over-the-counter (OTC) contract to buy or sell an asset at
a specified future date for a price agreed today. Unlike futures, forwards are not
standardized or exchange-traded, carrying counterparty risk. (Module 3: Instruments)

**Future** — A standardized contract traded on an exchange to buy or sell an asset at a
predetermined price on a specified future date. Futures are marked to market daily (gains
and losses settled each day). (Module 3: Instruments)

**FX Pair** — A foreign exchange instrument quoted as the ratio of two currencies. EUR/USD
= 1.10 means 1 euro = 1.10 US dollars. The first currency is the "base," the second is
the "quote." FX trades in pairs, never single currencies. (Module 3: Instruments)

**Maturity** — The date on which a bond's principal is repaid to the investor. A 10-year
Treasury bond issued today matures in 10 years. After maturity, no more coupons are paid.
`time_to_maturity = maturity_date - today`. (Module 3: Instruments)

**Option (Call)** — A contract giving the holder the right (not obligation) to BUY an
asset at the strike price before expiry. A call profits when the underlying price rises
above the strike. `payoff = max(S - K, 0)`. (Module 3: Instruments, Module 4: Pricing)

**Option (Put)** — A contract giving the holder the right (not obligation) to SELL an
asset at the strike price before expiry. A put profits when the underlying price falls
below the strike. `payoff = max(K - S, 0)`. (Module 3: Instruments, Module 4: Pricing)

**Par Value** — The face value of a bond, typically $1,000 or $100. Coupons are
calculated as a percentage of par. Bond prices are quoted as a percentage of par: a price
of 101.5 means the bond trades at 101.5% of par value. (Module 3: Instruments)

**Premium** — The price paid to purchase an options contract. The premium is the maximum
loss for the option buyer and the maximum gain for the option seller. Premium is
determined by the pricing model (Black-Scholes). (Module 4: Pricing Engine)

**Strike** — The price at which an option can be exercised. For a call option, the strike
is the price at which you can buy the underlying. In-the-money: S > K (call) or S < K
(put). At-the-money: S ≈ K. Out-of-the-money: S < K (call) or S > K (put).
(Module 3: Instruments, Module 4: Pricing)

**Swap** — A derivative contract in which two parties exchange cash flows. The most common
is an interest rate swap, where fixed-rate payments are exchanged for floating-rate
payments. Notional principal is not exchanged. (Module 3: Instruments)

**Yield** — The annualized return on a bond, expressed as a percentage. Current yield =
coupon / price. Yield to maturity (YTM) accounts for coupon payments, principal repayment,
and time value. When yield rises, bond price falls (inverse relationship).
(Module 3: Instruments)

---

## Risk Terms

**Beta** — A measure of a stock's volatility relative to the market. Beta = 1.0 means
the stock moves in line with the market. Beta > 1.0 means more volatile; Beta < 1.0 means
less volatile. `portfolio_beta = Σ(weight_i × beta_i)`. (Module 5: Risk Engine)

**Correlation** — A statistical measure (-1 to +1) of how two assets move relative to
each other. Correlation = +1 means they move together; -1 means they move oppositely.
Diversification relies on low or negative correlation between positions.
(Module 5: Risk Engine)

**CVaR (Conditional Value at Risk)** — Also called Expected Shortfall. The expected loss
given that the loss exceeds VaR. If 95% VaR is $1M, CVaR answers "when we lose more than
$1M, how much do we lose on average?" CVaR ≥ VaR, always. More conservative than VaR.
(Module 5: Risk Engine)

**Delta** — The rate of change of an option's price with respect to a $1 change in the
underlying asset's price. Delta ranges from 0 to 1 for calls, -1 to 0 for puts. ATM
options have delta ≈ 0.5. `position_delta = option_delta × quantity × contract_size`.
(Module 5: Risk Engine)

**Gamma** — The rate of change of delta with respect to underlying price. High gamma
means delta changes rapidly — the position's risk profile shifts quickly. Gamma is highest
for ATM options near expiry. `gamma_risk = 0.5 × gamma × (ΔS)²`.
(Module 5: Risk Engine)

**Hedging** — Taking an offsetting position to reduce risk. If you're long 100 shares of
AAPL, you might buy puts to hedge downside risk. A "delta-neutral" hedge adjusts positions
so that portfolio delta ≈ 0. (Module 5: Risk Engine)

**Mark-to-Market (MtM)** — Revaluing a position using current market prices rather than
the price at which it was originally traded. Daily MtM determines unrealized PnL. For
futures, MtM results in actual daily cash settlements. (Module 7: Position Tracker)

**PnL (Profit and Loss)** — The financial gain or loss on a position or portfolio.
Realized PnL = proceeds from closed positions minus cost basis. Unrealized PnL =
current value of open positions minus cost basis. `total_pnl = realized + unrealized`.
(Module 7: Position Tracker)

**Rho** — The sensitivity of an option's price to a 1% change in the risk-free interest
rate. Rho is typically small for short-dated options but significant for long-dated ones.
`rho_call = K × T × e^(-rT) × N(d2)`. (Module 5: Risk Engine)

**Scenario Analysis** — Evaluating portfolio performance under hypothetical market
conditions ("what if rates rise 200bp?", "what if equities drop 20%?"). Used alongside
VaR for stress testing. Each scenario applies a set of "shocks" to market factors.
(Module 5: Risk Engine)

**Stress Test** — An extreme scenario analysis using historically severe or hypothetical
market conditions (e.g., 2008 financial crisis, COVID crash). Regulators require banks to
prove they can survive stress scenarios. (Module 5: Risk Engine)

**Theta** — The rate of time decay of an option's price. Theta is typically negative —
options lose value as expiry approaches. Theta accelerates near expiry. For ATM options:
`theta ≈ -(S × σ × N'(d1)) / (2 × √T)`. (Module 5: Risk Engine)

**VaR (Value at Risk)** — The maximum expected loss over a given time period at a given
confidence level. "95% 1-day VaR of $1M" means there is a 95% chance the portfolio won't
lose more than $1M in a single day. Can be calculated via historical simulation, variance-
covariance, or Monte Carlo methods. (Module 5: Risk Engine)

**Vega** — The sensitivity of an option's price to a 1% change in implied volatility.
Long options have positive vega (benefit from volatility increase). Vega is highest for
ATM options with longer time to expiry. `vega = S × √T × N'(d1)`.
(Module 5: Risk Engine)

**Volatility (Historical)** — The annualized standard deviation of past returns.
Calculated from historical price data. `σ = std_dev(daily_returns) × √252`.
252 = trading days per year. (Module 5: Risk Engine)

**Volatility (Implied)** — The market's expectation of future volatility, backed out of
option prices using Black-Scholes. If an option trades at $5 and Black-Scholes gives $5
when σ = 25%, then implied vol = 25%. IV > historical vol suggests the market expects
future turbulence. (Module 4: Pricing Engine, Module 5: Risk Engine)

---

## Infrastructure Terms

**Algo Trading** — See Trading Terms section above. Cross-referenced here because it's
both a trading concept and an infrastructure capability. (Module 12: System Integration)

**Co-location** — Placing trading servers physically inside or next to the exchange's data
center. Reduces network latency from milliseconds to microseconds. Firms pay exchanges for
rack space. Our latency targets assume co-located deployment. (Module 6: Execution Gateway)

**Execution Management System (EMS)** — Software that routes orders to exchanges,
manages executions, and provides real-time fill reporting. Our ExecutionGateway module is a
simplified EMS. A full EMS handles multi-venue routing and algo execution.
(Module 6: Execution Gateway)

**FIX Protocol** — Financial Information eXchange protocol. The industry-standard
messaging format for electronic trading. Tag-value format: `35=D\x01` (tag 35 = message
type, value D = New Order Single). Our FIX parser extracts fields for order processing.
(Module 8: FIX Protocol)

**FPGA** — Field-Programmable Gate Array. Custom hardware programmed to process market
data and generate orders in nanoseconds, bypassing the CPU entirely. Used by the fastest
HFT firms. Our platform is CPU-based but targets similar algorithmic patterns.
(Module 13: Infrastructure)

**Market Data** — Real-time price and volume information broadcast by exchanges. Includes
quotes (bid/ask), trades (last price, volume), and order book updates. Our market data
handler (Module 9) parses and distributes this data. (Module 9: Market Data)

**Level 1 Data** — The best bid and best ask (top of book) plus last trade price. The
minimum information needed for basic trading. (Module 9: Market Data)

**Level 2 Data** — The full order book showing multiple price levels on each side with
aggregate quantity. Provides visibility into depth of liquidity. (Module 9: Market Data)

**Level 3 Data** — Individual order information in the book (every order ID, not just
aggregated levels). Only available to market makers and exchange members.
(Module 9: Market Data)

**Order Management System (OMS)** — Software that tracks orders through their lifecycle:
new → acknowledged → partially filled → filled / cancelled. Maintains order state and
provides audit trail. Our platform's order lifecycle spans Modules 2, 6, and 7.
(Module 2: Order Book, Module 6: Execution)

**Smart Order Router (SOR)** — Software that automatically routes orders to the venue
offering the best price, considering liquidity, fees, and latency. Required by best
execution regulations. In code, the SOR selects from multiple exchange connections.
(Module 6: Execution Gateway)

**Tick-to-Trade** — The end-to-end latency from receiving market data (tick) to sending
an order (trade). The primary performance metric for trading systems. Our target:
< 5 μs tick-to-trade. (Module 12: System Integration, Module 14: Benchmarks)

---

## Regulatory Terms

**AML (Anti-Money Laundering)** — Regulations requiring financial institutions to detect
and report suspicious transactions that may indicate money laundering. Trading platforms
must monitor for unusual patterns (e.g., rapid round-trip trades with no economic purpose).
(Module 16: Regulatory — if extended)

**Basel III** — International banking regulation framework requiring banks to maintain
minimum capital ratios and liquidity buffers. Affects risk management by mandating specific
VaR calculation methodologies and capital reserves. (Module 5: Risk Engine)

**Best Execution** — Regulatory requirement (MiFID II, SEC Rule 606) for brokers to
obtain the most favorable terms for client orders, considering price, cost, speed, and
likelihood of execution. Smart Order Routers implement best execution.
(Module 6: Execution Gateway)

**Circuit Breaker** — See Trading Terms section. Regulatory mechanism to halt trading
during extreme volatility. Exchange-level circuit breakers halt entire markets; per-stock
circuit breakers (LULD bands) halt individual securities. (Module 5: Risk Engine)

**Dodd-Frank** — US legislation (2010) imposing regulations on derivatives trading,
including mandatory clearing of standardized swaps, trade reporting to repositories, and
position limits. Affects how our system reports trades. (Module 11: Persistence)

**Fat Finger Check** — A pre-trade risk control that rejects orders with obviously
erroneous values — e.g., quantity 1,000,000 when the trader meant 1,000, or a price
10x the current market. Prevents costly mistakes. `if (qty > max_order_size) reject()`.
(Module 5: Risk Engine)

**KYC (Know Your Customer)** — Regulations requiring financial institutions to verify the
identity of clients before allowing them to trade. While KYC is handled upstream of our
platform, client IDs in orders link to KYC-verified accounts. (Module 16: Regulatory)

**MiFID II** — Markets in Financial Instruments Directive II (EU regulation). Requires
best execution, transaction reporting, transparency (pre- and post-trade), and data
recording. Trading platforms operating in EU markets must comply.
(Module 6: Execution Gateway, Module 11: Persistence)

**Position Limits** — Regulatory caps on the maximum position size a firm can hold in a
given instrument. Prevents market manipulation and excessive concentration risk. Our risk
engine checks position limits before accepting new orders.
(Module 5: Risk Engine)

**Trade Reporting** — Regulatory requirement to report executed trades to a trade
repository or regulator within specified timeframes (often seconds or minutes). Our
persistence layer generates reporting-ready trade records. (Module 11: Persistence)

---

## Yield Curve Terms

**Bootstrapping** — The mathematical process of extracting zero-coupon rates from
observed bond prices. Starting from the shortest maturity bond, each successive zero rate
is calculated using previously bootstrapped rates. This builds the complete zero curve
from market data. (Module 10: Yield Curve)

```
Given: 6M bond at 99.5 (coupon 2%), 1Y bond at 98.8 (coupon 3%)
Step 1: z_6m = -ln(99.5 / (100 + 1)) / 0.5 = ...
Step 2: Use z_6m to solve for z_1y from the 1Y bond price
Repeat for each maturity point
```

**Discount Factor** — The present value of $1 received at a future date. `DF(T) =
e^(-r × T)` for continuous compounding, where r is the zero rate for maturity T. Used to
value all future cash flows. A 5% rate for 1 year: `DF = e^(-0.05) ≈ 0.9512`.
(Module 10: Yield Curve)

**Forward Rate** — The interest rate for a future period implied by today's yield curve.
If the 1Y rate is 3% and the 2Y rate is 4%, the implied 1Y forward rate starting in 1
year is approximately 5%. `f(t1, t2) = (r2 × t2 - r1 × t1) / (t2 - t1)`.
(Module 10: Yield Curve)

**Yield Curve** — A graph plotting interest rates (yields) against maturities for bonds of
the same credit quality. A normal curve slopes upward (longer maturities = higher yields).
An inverted curve (short rates > long rates) often predicts recession. The curve is the
foundation of fixed-income pricing. (Module 10: Yield Curve)

**Zero Rate (Spot Rate)** — The yield on a zero-coupon bond for a given maturity. Zero
rates are the building blocks of the yield curve — all other rates (forward, par) can be
derived from them. `bond_price = Σ(coupon_i × DF(t_i)) + par × DF(T)` uses zero rates
through discount factors. (Module 10: Yield Curve)

---

## Quick Reference: Code ↔ Finance Mapping

| Finance Concept | C++ Representation | Module |
|----------------|-------------------|--------|
| Order | `struct Order { id, side, price, qty, type }` | M2 |
| Order Book | `std::map<Price, Level>` (bids desc, asks asc) | M2 |
| Fill | `struct Fill { order_id, price, qty, timestamp }` | M7 |
| Option | `struct Option { type, strike, expiry, underlying }` | M3 |
| Greeks | `struct Greeks { delta, gamma, vega, theta, rho }` | M5 |
| VaR | `double calculate_var(portfolio, confidence, horizon)` | M5 |
| FIX Message | `struct FixMessage { map<int, string_view> tags }` | M8 |
| Yield Curve | `std::vector<pair<double, double>>` (maturity, rate) | M10 |
| Position | `struct Position { symbol, qty, avg_price, pnl }` | M7 |
| Market Data | `struct Tick { symbol, bid, ask, last, volume, ts }` | M9 |
| Trade Report | WAL entry with CRC32 checksum | M11 |
| Config | `std::variant<int64_t, double, bool, string>` | M13 |
| Log Entry | `struct LogEntry { timestamp_ns, level, message }` | M13 |
| Timer Sample | `uint64_t nanoseconds` in `LatencyHistogram` | M13 |

---

## Cross-References

Every term in this glossary maps to one or more modules:

| Module | Key Terms |
|--------|-----------|
| Module 2: Order Book | Bid, Ask, Spread, Fill, IOC, FOK, GTC, Day Order, Tick Size, Lot Size, Iceberg, Stop Loss, Take Profit, Liquidity |
| Module 3: Instruments | Equity, Bond, Option, Future, Forward, Swap, CDS, FX Pair, Strike, Expiry, Coupon, Yield, Maturity, Par Value, Premium |
| Module 4: Pricing | Black-Scholes, Premium, Strike, Implied Volatility, Greeks |
| Module 5: Risk Engine | Delta, Gamma, Vega, Theta, Rho, VaR, CVaR, Beta, Correlation, Hedging, Stress Test, Scenario Analysis, Position Limits, Fat Finger, Leverage, Margin |
| Module 6: Execution | FIX Protocol, Slippage, Dark Pool, Smart Order Router, Co-location, EMS, Best Execution |
| Module 7: Position Tracker | PnL, Mark-to-Market, Fill, VWAP, Short Selling, Settlement |
| Module 8: FIX Protocol | FIX tags, message types, session management |
| Module 9: Market Data | Level 1/2/3, Tick, Market Data feed |
| Module 10: Yield Curve | Yield Curve, Discount Factor, Zero Rate, Forward Rate, Bootstrapping |
| Module 11: Persistence | Clearing, Settlement, Trade Reporting, WAL |
| Module 12: Integration | Tick-to-Trade, VWAP, TWAP, Algo Trading |
| Module 13: Infrastructure | FPGA, Arena allocator, Lock-free structures |
| Module 14: Benchmarks | Latency, Throughput, Percentiles |
