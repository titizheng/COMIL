
from models.basemodels import BasedMILTransformer as Instance_fuser
from models.functions import FusionHistoryFeatures,Memory,NCL_block 
from models.classmodel import ClassMultiMILTransformer as bag_aggregator
from utils.utils import make_parse


def create_model(args):
    
    basedmodel = Instance_fuser(args).cuda()
    classifymodel = bag_aggregator(args).cuda()
    Fusion_hispseudobag = FusionHistoryFeatures(args.embed_dim).cuda() 
    NCL_model = NCL_block(args.embed_dim, args.num_prototypes).cuda()
    memory = Memory()
    return basedmodel,classifymodel,memory,NCL_model,Fusion_hispseudobag


if __name__ == "__mian__":
    
    args = make_parse()
    create_model(args)
    
