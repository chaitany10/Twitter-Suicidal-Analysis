import quandl
import pandas as pd
api_key='xQoCVM_nP4haDtTy1CtP'
df = quandl.get("FMAC/HPI_TX", authtoken=api_key)
fiddy_states = pd.read_html("https://web-japps.ias.ac.in:8443/fellowship2019/lists/selectedList.jsp")
print(fiddy_states)
# for abbv in fiddy_states[0][1][1:]:
#     #print(abbv)
#     print("FMAC/HPI_"+str(abbv))