# Initialize a workbook
import string

import xlwt

from main import v

book = xlwt.Workbook()

# Add a sheet to the workbook
sheet1 = book.add_sheet("Sheet1")

# The data
# cols = ["A", "B", "C", "D", "E"]
cols = []
txt = v


cases = 10
if cases < 27:
    for uppercase in range(cases):
        cols.append(string.ascii_uppercase[uppercase])

# Loop over the rows and columns and fill in the values
for num in range(len(txt)):
      row = sheet1.row(num)
      for index, col in enumerate(cols):
          value = txt[num][index]
          row.write(index, value)

# Save the result
book.save("test.xls")