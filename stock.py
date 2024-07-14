import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import Tk, Label, Entry, Button, StringVar, OptionMenu, messagebox, Frame, Toplevel, Text, Scrollbar, VERTICAL, RIGHT, Y, END
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Extensive list of stock tickers for the autocomplete feature
STOCK_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "NFLX", "NKE", "NIO", "NCLH",
    "FB", "BABA", "V", "JPM", "JNJ", "WMT", "PG", "DIS", "MA", "HD", "PFE", "VZ", "BAC",
    "KO", "INTC", "CSCO", "CMCSA", "ADBE", "PEP", "XOM", "MRK", "ABT", "CVX", "T",
    "UNH", "MCD", "NVS", "LLY", "MDT", "NEE", "ORCL", "TMO", "HON", "TXN", "COST",
    "SNY", "SAP", "SBUX", "BMY", "QCOM", "CHTR", "LIN", "GILD", "LOW", "GE", "DHR",
    "MO", "IBM", "PM", "MMM", "AMT", "RTX", "ISRG", "AMGN", "BLK", "SPGI", "C",
    "DE", "LMT", "GS", "AXP", "PLD", "SYK", "MDLZ", "PYPL", "COP", "ADI", "INTU",
    "UNP", "BKNG", "ZTS", "DUK", "ADP", "CL", "USB", "MS", "CI", "TGT", "CVS",
    "EQIX", "EW", "BDX", "ETN", "TJX", "NSC", "MMC", "ITW", "FISV", "EOG", "CB",
    "NOC", "APD", "F", "SO", "HCA", "SHW", "CCI", "D", "AON", "SCHW", "PGR", "FIS",
    "CTSH", "KMB", "GM", "ICE", "MET", "MPC", "MCO", "AEP", "PSA", "DG", "DUK",
    "FDX", "TRV", "PNC", "VLO", "MCHP", "TT", "SRE", "DOV", "ROST", "ECL", "SPG",
    "GPN", "HLT", "CARR", "OTIS", "BAX", "HIG", "YUM", "AME", "EQR", "VRSK", "PSX",
    "PPG", "JCI", "APH", "ED", "STZ", "CME", "SYF", "AIG", "BKR", "IFF", "A", "HPQ",
    "EXC", "KHC", "DLTR", "EIX", "XEL", "AFL", "WY", "OMC", "HES", "LHX", "PPL",
    "CMS", "SWK", "PEG", "DTE", "LUV", "WEC", "SYY", "VTR", "FITB", "CERN", "ES",
    "LEN", "WELL", "GLW", "OKE", "FMC", "FRT", "HPE", "IP", "VIAC", "XRX", "VMC",
    "KMI", "K", "HSY", "FTV", "STT", "EMR", "WBA", "MOS", "BXP", "HUM", "ROL",
    "BUD", "IPG", "HBAN", "NUE", "RCL", "CPRT", "FAST", "MKC", "L", "GPC", "CLX",
    "TDG", "CTXS", "WDC", "DGX", "UDR", "WHR", "VFC", "NDAQ", "MSI", "ROK", "UAL",
    "XRAY", "ULTA", "KSS", "HAS", "HRL", "PKG", "CF", "SBAC", "TER", "TTWO", "QRVO",
    "RSG", "DRI", "CHD", "DISH", "HWM", "ETR", "SIVB", "KEYS", "VTRS", "NLSN", "MLM",
    "RE", "BBY", "WAT", "EFX", "ZBRA", "LW", "LVS", "CNP", "REGN", "PH", "NTRS",
    "LDOS", "MKTX", "STE", "PKI", "JBHT", "MAS", "CTLT", "XYL", "LKQ", "RJF", "EVRG",
    "TROW", "ETSY", "WMB", "MHK", "AEE", "FDS", "BIO", "UDR", "KEY", "JBHT", "COO",
    "MCK", "AES", "NVR", "TSCO", "MAA", "AAP", "BMRN", "CBOE", "CE", "CEG", "CHRW",
    "CINF", "CMS", "COG", "CPT", "EXPD", "FERG", "FFIV", "HII", "IRM", "IVZ",
    "JWN", "MAS", "MRO", "NDSN", "NLOK", "NWL", "OKE", "PEAK", "PEP", "PKG", "PNW",
    "REG", "RHI", "RMD", "RYN", "SNA", "SYY", "TEG", "TRIP", "TROW", "TXT",
    "UAL", "URI", "VFC", "VLO", "VNO", "WAB", "WAT", "WEC", "WELL", "WFC",
    "WY", "XLNX", "YUM", "ZBH", "ZTS"
]

class AutoCompleteCombobox(ttk.Combobox):
    def set_completion_list(self, completion_list):
        self._completion_list = sorted(completion_list)
        self._hits = []
        self._hit_index = 0
        self.position = 0
        self.bind('<KeyRelease>', self.handle_keyrelease)
        self['values'] = self._completion_list

    def autocomplete(self, delta=0):
        if delta:
            self.delete(self.position, END)
        else:
            self.position = len(self.get())
        
        _hits = []
        for item in self._completion_list:
            if item.startswith(self.get().upper()):
                _hits.append(item)

        if _hits != self._hits:
            self._hit_index = 0
            self._hits = _hits

        if _hits:
            self._hit_index = (self._hit_index + delta) % len(_hits)
            self.delete(0, END)
            self.insert(0, _hits[self._hit_index])
            self.select_range(self.position, END)

    def handle_keyrelease(self, event):
        if event.keysym in ('BackSpace', 'Left', 'Right', 'Up', 'Down'):
            return
        self.autocomplete()

def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

def analyze_data(data, analysis_type, window=None):
    if analysis_type == "Summary":
        return data.describe()
    elif analysis_type == "Plot":
        fig, ax = plt.subplots()
        data['Close'].plot(title='Stock Closing Prices', ax=ax, legend=True)
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        ax.grid(True)
        return fig
    elif analysis_type == "Moving Average":
        if window is not None:
            data['Moving Average'] = data['Close'].rolling(window=window).mean()
            fig, ax = plt.subplots()
            data[['Close', 'Moving Average']].plot(title='Stock Closing Prices and Moving Average', ax=ax, legend=True)
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.grid(True)
            return fig
        else:
            return "Moving average window is required for moving average analysis."
    elif analysis_type == "Predict":
        return predict_stock_prices(data)
    else:
        return "Invalid analysis type. Please choose Summary, Plot, Moving Average, or Predict."

def predict_stock_prices(data):
    data = data[['Close']].dropna()
    data['Prediction'] = data['Close'].shift(-30)
    
    X = data.drop(['Prediction'], axis=1)[:-30]
    y = data['Prediction'][:-30]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    future = data.drop(['Prediction'], axis=1)[-30:]
    future_predictions = model.predict(future)
    
    fig, ax = plt.subplots()
    ax.plot(data.index[-60:], data['Close'][-60:], label='Historical Prices')
    ax.plot(future.index, future_predictions, label='Predicted Prices', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()
    ax.grid(True)
    
    mse = mean_squared_error(y_test, predictions)
    return fig, mse

def on_submit():
    ticker = ticker_entry.get()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()
    analysis_type = analysis_type_var.get()
    
    if not ticker or not start_date or not end_date or analysis_type == "Select Analysis":
        messagebox.showerror("Input Error", "Please fill all the fields and select a valid analysis type.")
        return
    
    data = get_stock_data(ticker, start_date, end_date)
    
    if analysis_type == "Moving Average":
        window = window_entry.get()
        if not window:
            messagebox.showerror("Input Error", "Please enter a valid window size for Moving Average.")
            return
        window = int(window)
        result = analyze_data(data, analysis_type, window)
    else:
        result = analyze_data(data, analysis_type)
    
    if isinstance(result, str):
        messagebox.showinfo("Result", result)
    elif isinstance(result, tuple) and analysis_type == "Predict":
        fig, mse = result
        display_plot(fig)
        messagebox.showinfo("Prediction MSE", f"Mean Squared Error of the predictions: {mse}")
    elif isinstance(result, pd.DataFrame):
        display_summary(result)
    else:
        display_plot(result)

def display_summary(summary):
    summary_window = Toplevel(root)
    summary_window.title("Stock Data Summary")
    
    summary_text = tabulate(summary, headers='keys', tablefmt='fancy_grid')
    text = Text(summary_window, wrap='word', width=80, height=20)
    text.insert(END, summary_text)
    text.pack()
    
    scrollbar = Scrollbar(summary_window, orient=VERTICAL, command=text.yview)
    scrollbar.pack(side=RIGHT, fill=Y)
    text.config(yscrollcommand=scrollbar.set)

def display_plot(fig):
    plot_window = Toplevel(root)
    plot_window.title("Stock Data Plot")
    
    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Set up GUI
root = Tk()
root.title("Stock Analysis Tool")

style = ttk.Style()
style.configure('TButton', font=('Helvetica', 12), padding=10)
style.configure('TLabel', font=('Helvetica', 12))
style.configure('TEntry', font=('Helvetica', 12))
style.configure('TOptionMenu', font=('Helvetica', 12))

frame = Frame(root, padx=10, pady=10)
frame.pack(padx=10, pady=10)

Label(frame, text="Stock Ticker:").grid(row=0, column=0, sticky='w')
ticker_entry = AutoCompleteCombobox(frame)
ticker_entry.set_completion_list(STOCK_TICKERS)
ticker_entry.grid(row=0, column=1, padx=5, pady=5)

Label(frame, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0, sticky='w')
start_date_entry = ttk.Entry(frame)
start_date_entry.grid(row=1, column=1, padx=5, pady=5)

Label(frame, text="End Date (YYYY-MM-DD):").grid(row=2, column=0, sticky='w')
end_date_entry = ttk.Entry(frame)
end_date_entry.grid(row=2, column=1, padx=5, pady=5)

analysis_type_var = StringVar(frame)
analysis_type_var.set("Select Analysis")
option_menu = ttk.OptionMenu(frame, analysis_type_var, "Select Analysis", "Summary", "Plot", "Moving Average", "Predict")
option_menu.grid(row=3, column=0, columnspan=2, sticky='we', padx=5, pady=5)

window_label = ttk.Label(frame, text="Moving Average Window (for MA only):")
window_label.grid(row=4, column=0, sticky='w')
window_entry = ttk.Entry(frame)
window_entry.grid(row=4, column=1, padx=5, pady=5)

submit_button = ttk.Button(frame, text="Submit", command=on_submit)
submit_button.grid(row=5, column=0, columnspan=2, pady=10)

root.mainloop()
