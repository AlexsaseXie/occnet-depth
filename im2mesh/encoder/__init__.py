from im2mesh.encoder import (
    conv, depth_conv, pix2mesh_cond, pointnet, pointnet2,
    psgn_cond, r2n2, voxels,
)


encoder_dict = {
    'simple_conv': conv.ConvEncoder,
    'resnet18': conv.Resnet18,
    'resnet34': conv.Resnet34,
    'resnet50': conv.Resnet50,
    'resnet101': conv.Resnet101,
    'r2n2_simple': r2n2.SimpleConv,
    'r2n2_resnet': r2n2.Resnet,
    'pointnet_simple': pointnet.SimplePointnet,
    'pointnet_resnet': pointnet.ResnetPointnet,
    'psgn_cond': psgn_cond.PCGN_Cond,
    'voxel_simple': voxels.VoxelEncoder,
    'pixel2mesh_cond': pix2mesh_cond.Pix2mesh_Cond,

    'depth_resnet18': depth_conv.Depth_Resnet18,
    'pointnet': pointnet.PointNetEncoder,
    'pointnet_res': pointnet.PointNetResEncoder,
    'stacked_pointnet': pointnet.StackedPointnet,
    'msn_pointnet': pointnet.MSNPointNetFeat,
    'pointnet2': pointnet2.PointNet2SSGEncoder,
    'pointnet2_ssg': pointnet2.PointNet2SSGEncoder,
    'pointnet2_msg': pointnet2.PointNet2MSGEncoder,
}
