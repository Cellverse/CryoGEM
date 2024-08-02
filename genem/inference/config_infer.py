import logging

logger = logging.getLogger(__name__)

dataset_list = [
    "Proteasome(10025)", 
    "Ribosome(10028)", 
    "Integrin(10345)", 
    "PhageMS2(10075)", 
    "HumanBAF(10590)"
]

weight_map_dict = {
    "Proteasome(10025)": "testing/data/Proteasome(10025)/weightmaps",
    "Ribosome(10028)":   "testing/data/Ribosome(10028)/weightmaps",
    "Integrin(10345)":   "testing/data/Integrin(10345)/weightmaps",
    "PhageMS2(10075)":   "testing/data/PhageMS2(10075)/weightmaps",
    "HumanBAF(10590)":   "testing/data/HumanBAF(10590)/weightmaps",
}

model_dict = {
    "Proteasome(10025)": "testing/checkpoints/Proteasome(10025)/200_net_G.pth",
    "Ribosome(10028)":   "testing/checkpoints/Ribosome(10028)/200_net_G.pth",
    "Integrin(10345)":   "testing/checkpoints/Integrin(10345)/200_net_G.pth",
    "PhageMS2(10075)":   "testing/checkpoints/PhageMS2(10075)/200_net_G.pth",
    "HumanBAF(10590)":   "testing/checkpoints/HumanBAF(10590)/200_net_G.pth",
}

adaptor_dict = {
    "Proteasome(10025)": "testing/checkpoints/Proteasome(10025)/200_net_F.pth",
    "Ribosome(10028)":   "testing/checkpoints/Ribosome(10028)/200_net_F.pth",
    "Integrin(10345)":   "testing/checkpoints/Integrin(10345)/200_net_F.pth",
    "PhageMS2(10075)":   "testing/checkpoints/PhageMS2(10075)/200_net_F.pth",
    "HumanBAF(10590)":   "testing/checkpoints/HumanBAF(10590)/200_net_F.pth",
}

apix_dict = {
    "Proteasome(10025)": 4.62,
    "Ribosome(10028)":   5.36,
    "Integrin(10345)":   4.035,
    "PhageMS2(10075)":   4.64,
    "HumanBAF(10590)":   4.500,
}
    

def get_template(dataset):
    return f"save_images/template_clean_mrc_clean/{dataset}.clean.template.mrc"

def get_template_mask(dataset):
    return f"save_images/template_clean_mrc_clean/{dataset}.clean.template.mask.png"
    

clean_mrc_dict = {
    "Proteasome(10025)": get_template("Proteasome(10025)"),
    "Ribosome(10028)":   get_template("Ribosome(10028)"),
    "Integrin(10345)":   get_template("Integrin(10345)"),
    "PhageMS2(10075)":   get_template("PhageMS2(10075)"),
    "HumanBAF(10590)":   get_template("HumanBAF(10590)"),
}

clean_mrc_mask_dict = {
    "Proteasome(10025)": get_template_mask("Proteasome(10025)"),
    "Ribosome(10028)":   get_template_mask("Ribosome(10028)"),
    "Integrin(10345)":   get_template_mask("Integrin(10345)"),
    "PhageMS2(10075)":   get_template_mask("PhageMS2(10075)"),
    "HumanBAF(10590)":   get_template_mask("HumanBAF(10590)"),
}


template_particle_num_dict = {
    "Proteasome(10025)":  900,
    "Ribosome(10028)":    120,
    "Integrin(10345)":    60,
    "PhageMS2(10075)":    90,
    "HumanBAF(10590)":    210,
}

template_mask_ratio_dict = {
    "Proteasome(10025)":  0.75,
    "Ribosome(10028)":    0.5,
    "Integrin(10345)":    0.4,
    "PhageMS2(10075)":    0.5,
    "HumanBAF(10590)":    0.55,  
}

symmetry_dict = {
    "Proteasome(10025)":  "D7",
    "Ribosome(10028)":    "C1",
    "Integrin(10345)":    "C1",
    "PhageMS2(10075)":    "D7",
    "HumanBAF(10590)":    "C1",
}
