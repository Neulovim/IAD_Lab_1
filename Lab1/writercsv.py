# Initialize a workbook
import string

import xlwt

from Lab1.main import v, w

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

txt = w
row = sheet1.row(len(txt)+1)
for index, col in enumerate(cols):
    value = txt[index]
    row.write(index, value)

# Save the result
book.save("numbers.xls")