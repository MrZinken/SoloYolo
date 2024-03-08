dividend = 9660
for x in range(641):  # x should be less than 640
    if x != 0 and dividend % 16*(640 - x) == 0:
        print(f"x = {x}, (640 - x) is a divisor of {dividend}")