import os
from typing import List


def get_file_list(
    root_dir: str,
    absolute_path: bool = True,
    file_ending: bool = True,
    file_type_filter: str = None,
) -> List:

    assert os.path.exists(root_dir)
    list_of_data_locs = []
    for (root_dir, dirname, filename) in os.walk(root_dir):
        for file in filename:
            if file_type_filter is not None and file_type_filter in file:
                if not file_ending:
                    file = file[: file.index(".")]
                if absolute_path:
                    list_of_data_locs.append(os.path.join(root_dir, file))
                else:
                    list_of_data_locs.append(file)
    return sorted(list_of_data_locs)


def get_model_file_list_for_two_domain_experiment(experiment_dir:str, domain_names:List[str], n_folds:int, use_clf:bool)->dict:
    domain_models_i = []
    domain_models_j = []
    latent_dcm_models = []
    latent_clf_models = []
    for i in range(n_folds):
        # domain_models_i.append(experiment_dir +'/fold_{}/final_{}_model.pth'.format(i, domain_names[0]))
        # domain_models_j.append(experiment_dir + '/fold_{}/final_{}_model.pth'.format(i, domain_names[1]))
        # latent_dcm_models.append(experiment_dir + '/fold_{}/final_dcm.pth'.format(i))
        # if use_clf:
        #     latent_clf_models.append(experiment_dir + '/fold_{}/final_clf.pth'.format(i))

        domain_models_i.append(experiment_dir + '/fold_{}/epoch_950/model_{}.pth'.format(i, domain_names[0]))
        domain_models_j.append(experiment_dir + '/fold_{}/epoch_950/model_{}.pth'.format(i, domain_names[1]))
        latent_dcm_models.append(experiment_dir + '/fold_{}/epoch_950/dcm.pth'.format(i))
        if use_clf:
            latent_clf_models.append(experiment_dir + '/fold_{}/epoch_950/clf.pth'.format(i))
    if len(latent_clf_models) == 0:
        latent_clf_models = None

    model_locations_dict = {'domain_models_i':domain_models_i, 'domain_models_j':domain_models_j,
                 'latent_dcm_models':latent_dcm_models, "latent_clf_models":latent_clf_models}

    return model_locations_dict

