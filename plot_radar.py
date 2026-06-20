import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


categories = ['Accuracy', 'Sensitivity (Recall)', 'Mathematical Rigor', 
              'Explainability (XAI)', 'Reliability (Uncertainty)']

baseline = [91, 88, 30, 40, 20] 
ours = [86, 100, 95, 90, 90]     

label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(baseline))

plt.figure(figsize=(10, 10))
plt.subplot(polar=True)


plt.plot(label_loc, baseline, label='Standard CNN (Baseline)', color='#e74c3c', linewidth=2)
plt.fill(label_loc, baseline, color='#e74c3c', alpha=0.1)


plt.plot(label_loc, ours, label='Proposed Quadrature-Enhanced Model', color='#3498db', linewidth=3)
plt.fill(label_loc, ours, color='#3498db', alpha=0.2)

lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories)
plt.title('Multi-dimensional Performance Comparison', size=15, fontweight='bold', y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))


plt.savefig('results/radar_comparison.png', dpi=300, bbox_inches='tight')
plt.show()