import re;
import lightgbm as lgb;
from matplotlib import pyplot as plt;


model = lgb.Booster(model_file="HyScreen_LSA_SBERT_v3.model");
generic_names = model.feature_name();
print("Old names:", generic_names);

feature_names = [
    "LSA-Resume-Score",
    "SBERT-Job_Role-Score",
    "SBERT-Job_Description-Score",
    "SBERT-Job_Education-Score"
]; #Note: Spaces not allowed in name

model_text = model.model_to_string();
for old, new in zip(generic_names, feature_names):
    model_text = re.sub(r"\b" + re.escape(old) + r"\b", new, model_text); #regex my beloved

patched_model = lgb.Booster(model_str=model_text);
print("New names:", patched_model.feature_name());

ax = lgb.plot_tree(
    patched_model,
    tree_index=0,    
	dpi = 400,
	orientation = 'vertical',
    show_info=["split_gain", "internal_value", "internal_count", "internal_weight","leaf_count", "leaf_weight", "data_percentage"]
); #See more: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.plot_tree.html

plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0);

#Save as image
filename = "tree_export.png";
#plt.savefig(filename, bbox_inches='tight', pad_inches=0);
plt.savefig(filename, dpi=1200, bbox_inches='tight', pad_inches=0);
print("Save file as: "+filename);

#Show in TKinter window
plt.show(); 
