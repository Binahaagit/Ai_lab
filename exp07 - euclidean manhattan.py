x1 = int(input("Enter x1: "))
y1 = int(input("Enter y1: "))

x2 = int(input("Enter x2: "))
y2 = int(input("Enter y2: "))

euclidean = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5

manhattan = abs(x2 - x1) + abs(y2 - y1)

print("Euclidean distance:", euclidean)
print("Manhattan distance:", manhattan)

