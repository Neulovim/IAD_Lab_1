import xlrd, xlwt

rb = xlrd.open_workbook('test.xlsx')
sheet = rb.sheet_by_index(0)
vals = [sheet.row_values(rownum) for rownum in range(sheet.nrows)]
print(vals)