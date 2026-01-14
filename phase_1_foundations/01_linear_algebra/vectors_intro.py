import numpy as np
import matplotlib.pyplot as plt

print("=== 1. VECTOR INTRODUCTION FOR LINEAR ALGEBRA ===")
print("Vectors = arrows from origin representing position/direction/magnitude\n")

# Vector as NumPy array (coordinates)
v1 = np.array([3, 4])        # 2D vector (x=3, y=4)
v2 = np.array([-1, 2, 5])    # 3D vector (x=-1, y=2, z=5)

print(f"2D Vector v1: {v1} (points 3 right, 4 up)")
print(f"3D Vector v2: {v2} (points -1 right, 2 up, 5 forward)")
print(f"Magnitude of v1: {np.linalg.norm(v1):.2f}")  # Length = sqrt(3²+4²)=5
print()

# ========================================
print("=== 2. REAL COORDINATE SPACES (R^n) ===")
print("R^n = n-dimensional space. R^2=plane, R^3=3D space, R^784=embedding space\n")

# R^1: Real line (1D)
R1_points = np.array([ -3, -1, 0, 2, 5 ])
print("R^1 examples:", R1_points)

# R^2: Plane with standard basis vectors e1=(1,0), e2=(0,1)
e1, e2 = np.array([1, 0]), np.array([0, 1])
point_A = 3*e1 + 2*e2  # Any point = linear combination of basis
print(f"R^2: Point A = 3*e1 + 2*e2 = {point_A}")

# R^3: 3D space with basis e1,e2,e3
e1_3d, e2_3d, e3_3d = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
point_B = 1*e1_3d + 4*e2_3d + 2*e3_3d
print(f"R^3: Point B = 1*e1 + 4*e2 + 2*e3 = {point_B}")

# ML Connection: Dataset = matrix where each COLUMN is a vector in R^n
dataset = np.column_stack([v1, point_A[:2], point_B[:2]])
print(f"\nDataset matrix (3 samples in R^2):\n{dataset}")

# ========================================
# VISUALIZATION (R^2 only, since matplotlib is 2D)
print("\n=== VISUALIZATION (Opening plot window) ===")
plt.figure(figsize=(14, 5))  # Create figure window 14 inches wide, 5 tall

# Plot 1: Vectors as arrows
plt.subplot(1, 3, 1)  # Create 1 row, 3 columns of plots; this is plot #1
origin = [0, 0]  # Starting point for all vectors
plt.quiver(*origin, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, 
           color='red', width=0.015, label=f'v1 = {v1}')
plt.quiver(*origin, 2, 1, angles='xy', scale_units='xy', scale=1, 
           color='blue', width=0.015, label='v2 = [2,1]')
plt.xlim(-2, 5)  # x-axis from -2 to 5
plt.ylim(-2, 5)  # y-axis from -2 to 5
plt.grid(True, alpha=0.3)  # Show grid, 30% transparency
plt.legend()
plt.title('Vectors as Arrows (R²)')
plt.axis('equal')  # Equal scaling for x and y axes 

# Plot 2: Basis vectors + linear combinations
plt.subplot(1, 3, 2)  # Create 1 row, 3 columns of plots; this is plot #2
origin = [0, 0]  # Starting point for all vectors
plt.quiver(*origin, e1[0], e1[1], angles='xy', scale_units='xy', scale=1, 
           color='red', width=0.015, label='e₁ = (1,0)')
plt.quiver(*origin, e2[0], e2[1], angles='xy', scale_units='xy', scale=1, 
           color='green', width=0.015, label='e₂ = (0,1)')
plt.quiver(*origin, point_A[0], point_A[1], angles='xy', scale_units='xy', scale=1, 
           color='blue', width=0.008, label=f'3e₁ + 2e₂ = {point_A}')
plt.xlim(-1, 4)  # x-axis from -1 to 4
plt.ylim(-1, 3)  # y-axis from -1 to 3
plt.grid(True, alpha=0.3)  # Show grid, 30% transparency
plt.legend()
plt.title('Coordinate Space R²\n(Basis + Linear Combination)')
plt.axis('equal')  # Equal scaling for x and y axes

# Plot 3: Dataset as points
plt.subplot(1, 3, 3)  # Third plot in the 1x3 grid
plt.scatter(dataset[0, :], dataset[1, :], s=100, c=['red', 'blue', 'green'], 
            edgecolors='black', linewidth=2)
for i in range(dataset.shape[1]):  # Loop through 3 columns (samples)
    plt.annotate(f'v{i+1}', (dataset[0, i], dataset[1, i]), xytext=(5, 5), textcoords='offset points')
plt.grid(True, alpha=0.3)
plt.title('ML Dataset\n(Points in R²)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

# ========================================
print("\n=== ML CONNECTIONS ===")
print("1. Every neural net input = vector in R^n")
print("2. Every neural net weight = matrix (transforms R^n → R^m)")
print("3. Dataset = matrix where columns = vectors in R^n")
print("4. Forward pass = matrix multiplication: y = W @ x")
print("\n✅ Concepts coded! Commit to GitHub:")
print("git add vectors_intro.py")
print("git commit -m 'Week 1: Vectors + R^n spaces with visualization'")
print("git push")
