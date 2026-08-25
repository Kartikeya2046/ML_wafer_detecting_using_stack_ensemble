import pickle
from sklearn.metrics import classification_report, precision_recall_fscore_support

def main():
    print("Loading results...")
    with open(r"d:\assignment college\VLSI_PROJECT\data\stacking_outputs.pkl", 'rb') as f:
        stacking_outputs = pickle.load(f)
        
    with open(r"d:\assignment college\VLSI_PROJECT\data\mfe_fnn_outputs.pkl", 'rb') as f:
        mfe_outputs = pickle.load(f)
        
    with open(r"d:\assignment college\VLSI_PROJECT\data\cnn_outputs.pkl", 'rb') as f:
        cnn_outputs = pickle.load(f)
        
    y_test = stacking_outputs['y_test']
    
    y_pred_mfe = mfe_outputs['test_prob'].argmax(axis=1)
    y_pred_cnn = cnn_outputs['test_prob'].argmax(axis=1)
    y_pred_stack = stacking_outputs['test_prob'].argmax(axis=1)
    
    target_names = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full', 'none']
    
    # Calculate per-class metrics
    p_mfe, r_mfe, f1_mfe, _ = precision_recall_fscore_support(y_test, y_pred_mfe)
    p_cnn, r_cnn, f1_cnn, _ = precision_recall_fscore_support(y_test, y_pred_cnn)
    p_stk, r_stk, f1_stk, _ = precision_recall_fscore_support(y_test, y_pred_stack)
    
    markdown_output = "# Model Comparison Results\n\n"
    markdown_output += "## Per-Class F1 Score Comparison\n\n"
    markdown_output += "| Class | MFE+FNN | CNN | Stacking Ensemble |\n"
    markdown_output += "|-------|---------|-----|-------------------|\n"
    
    for i, name in enumerate(target_names):
        markdown_output += f"| {name} | {f1_mfe[i]:.4f} | {f1_cnn[i]:.4f} | {f1_stk[i]:.4f} |\n"
        
    markdown_output += f"| **Macro Avg** | **{mfe_outputs['f1_macro']:.4f}** | **{cnn_outputs['f1_macro']:.4f}** | **{stacking_outputs['f1_macro']:.4f}** |\n"
    
    markdown_output += "\n## Discussion\n"
    markdown_output += "The paper reported F1-macro scores around 0.8599 (MFE), 0.8679 (CNN), and 0.8949 (Stacking-MLR) at N=162,946. "
    markdown_output += "If our scores are roughly within standard deviations (or differ, possibly due to exact train/test indices, class-weight differences, or the FNN meta-learner substituting the MLR model), we document it here.\n\n"
    markdown_output += f"Our MFE F1: {mfe_outputs['f1_macro']:.4f}\n\n"
    markdown_output += f"Our CNN F1: {cnn_outputs['f1_macro']:.4f}\n\n"
    markdown_output += f"Our Stacking F1: {stacking_outputs['f1_macro']:.4f}\n\n"
    
    print(markdown_output)
    
    with open(r"d:\assignment college\VLSI_PROJECT\results_comparison.md", 'w') as f:
        f.write(markdown_output)
        
    print("Phase 6 comparison saved to results_comparison.md!")

if __name__ == "__main__":
    main()
