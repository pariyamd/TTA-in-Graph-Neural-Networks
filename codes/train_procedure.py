import torch
import os
from torch import nn
import logging
from model.model_utilies import save_model, load_model, test, process_adj, generate_normalized_adjs, generate_one_hot_label, predict
import torch.nn.functional as F
from homottt import HomoTTT
import copy
from gapgc import NodeGAPGC, EdgeMaskingAugmenter,test_time_node_adaptation
import pickle
def train_procedure(args, logger, model, source_optimizer, target_optimizer, criterion, source_data, target_data, target_structure_data, structure_adj):
    # training
    best_val_acc = 0.0
    best_test_acc = 0.0
    best_val_loss = 1e8

    args.logger.info("source period")
    if args.is_source_train:
        for epoch in range(args.source_epochs):
            train_loss, train_accuracy = model.train_source(source_data, source_optimizer, criterion, epoch)
            val_loss, val_accuracy = test(model, args, source_data, criterion, 'valid')
            # micro, macro = test(model, args, source_data, criterion, 'valid')
            args.logger.info('Epoch\t{:03d}\ttrain:acc\t{:.6f}\tcross_entropy\t{:.6f}\tvalid:acc\t{:.6f}\tcross_entropy\t{:.6f}'.format(
            epoch, train_accuracy, train_loss, val_accuracy, val_loss))
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_val_loss = val_loss
                save_model(args, "source", model)
        args.logger.info('Best valid acc\t{:.6f}\t Best valid loss\t{:.6f}'.format(best_val_acc, best_val_loss))

        args.logger.info(f"loss and acc before adapting {test(model, args, target_data, criterion, 'test')}")
    else:
        model = load_model(args, "source", model)

    args.logger.info("target period")
    if args.is_baseline:
        best_test_acc = test(model, args, target_data, criterion, "test")

    if args.adaptation_method == "graphtta":
        adaptation_model = NodeGAPGC(
            gnn_model=model,
            augmenter=EdgeMaskingAugmenter( gnn=copy.deepcopy(model), temperature=0.05),
            hidden_dim=128)
        
        encoder_optimizer = torch.optim.Adam(adaptation_model.gnn.parameters(), lr=0.001)
        augmenter_optimizer = torch.optim.Adam(adaptation_model.augmenter.parameters(), lr=0.005)
        metrics = test_time_node_adaptation(
            model=adaptation_model,
            data=target_data,
            encoder_optimizer=encoder_optimizer,
            augmenter_optimizer=augmenter_optimizer,
            val_mask=target_data.val_mask,
            test_mask=None,
            epochs=args.target_epochs,
            log_every=1,
            projection = False,
            args=args,
            logger=logger
            )
        
    if args.adaptation_method == "homottt":

        # Initialize HomoTTT
        homottt = HomoTTT(
            model=model,
            num_clusters=args.num_label,
            drop_edge_ratio=0.2,
            update_layers=4,
            seed=args.random_seed
        )

        # optimizer = torch.optim.Adam(homottt.model.parameters(), lr=0.001)

        adapted_model = homottt.test_time_train(
            data=target_data, 
            optimizer=target_optimizer,
            num_iterations=args.target_epochs,
            args=args,
            logger=logger
        )


    if args.adaptation_method == "soga":
        model = load_model(args, "source", model)
        micro, macro = test(model, args, target_data, criterion, 'test')
        print("\n\n\n")
        print(micro,macro)
        print("\n\n\n")
        return micro, macro

    #     model.init_target(target_structure_data, target_data)

    #     for epoch in range(args.target_epochs):
    #         test_accuracy = model.train_target(target_data, target_structure_data,structure_adj, criterion, target_optimizer, epoch)
    #         # args.logger.info('Epoch\t{:03d}\ttest:acc\t{:.6f}'.format(epoch, test_accuracy))
    #         micro, macro = test(model, args, target_data, criterion, 'test')
    #         args.logger.info("{:.6f}".format(micro), key="Micro F1")
    #         args.logger.info("{:.6f}".format(macro), key="Macro F1")
           
    #         if test_accuracy > best_test_acc:
    #             best_test_acc = test_accuracy
    #             save_model(args, "target", model)
            
    #     args.logger.info('Best test acc\t{:.6f}'.format(best_test_acc))

    # print(len(logger.values["Loss"]))
    # with open("./record/" + args.model_name+f"/{args.adaptation_method}_{args.random_seed}.pkl", 'wb') as f:
    #     pickle.dump(logger.values, f)
    # handlers = logger.logger.handlers[:]
    # for handler in handlers:
    #     handler.close()
    #     logger.logger.removeHandler(handler)

    # return best_test_acc
