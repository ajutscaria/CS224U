file = open('log','r')
basic = open('basic.txt', 'w')
extended = open('extended.txt', 'w')

readingbasic = False;
readingextended = False;
count = 0
for line in file:
    line = line.strip()
    
    if "Example " in line:
        count = count + 1
    
    if readingbasic:
        basic.write(line + "\n")
    if readingextended:
        extended.write(line + "\n")
    
    if line == "}":
        count = count - 1;
    
    if count == -1:
        readingbasic = False;
        readingextended = False;
    
    if line == "Extended trigger prediction {":
        readingextended = True
        count = 0
    
    if line == "Basiccc trigger prediction {":
        readingbasic = True
        count = 0

basic.close()
extended.close()