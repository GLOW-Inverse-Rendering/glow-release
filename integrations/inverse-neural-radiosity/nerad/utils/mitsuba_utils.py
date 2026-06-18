import re
from pathlib import Path

import drjit as dr
import mitsuba as mi

import torch.nn.functional as F
import tempfile

@dr.wrap_ad(source="drjit", target="torch")
def block_sum_image(img, block_y, block_x):
    img = img.permute(2, 0, 1) #C x H x W
    img = img.unsqueeze(dim=0) # 1x C x H x W
    img = F.avg_pool2d(img, kernel_size=(block_y, block_x))
    return img.squeeze(dim=0).permute(1, 2, 0) # H x W x C


def float_to_tens_safe(vec):
    # Converts a Vector3f to a TensorXf safely in mitsuba while keeping the gradients;
    # a regular type cast mi.TensorXf(vector) detaches the gradients
    return mi.TensorXf(dr.ravel(vec), shape=(dr.shape(vec)[0],))


def vec_to_tens_safe(vec):
    # Converts a Vector3f to a TensorXf safely in mitsuba while keeping the gradients;
    # a regular type cast mi.TensorXf(vector) detaches the gradients
    return mi.TensorXf(dr.ravel(vec), shape=(dr.shape(vec)[1], dr.shape(vec)[0]))


def load_scene_with_edits(file, shape_mode: str, brdf_mode: dict, emitter_mode: str) -> mi.Scene:
    """Edit XML and assign one single SVBRDF to all shapes in the scene"""


    # print("======temporarily disabling all edits=====")
    # return mi.load_file(file)
    if True or   ('irb' in file or 'custom_kitch' in file or 'cornell-box-nobox' in file or 'cornell-box' in file or 'staircase' in file or 'living-room-2' in file or 'veach_ajar' in file or 'nerf_scenes' in file or 'cube' in file or 'bunny' in file):
        file = Path(file)
        scene_txt = (file).read_text("utf-8")

        shapes_file = file.parent/'shapes.xml'
        shape_txt = (shapes_file).read_text("utf-8")
        emitters_file = file.parent/'emitters.xml'
        emitters_txt = (emitters_file).read_text("utf-8")

        if shape_mode != 'same':
            assert brdf_mode["name"] != 'same'
            results = shape_mode.split(",")
            if len(results) == 1:
                shape_mode = results[0]
                assert shape_mode == "none"
                shape_txt = ""

            elif len(results) == 2:
                shape_mode,custom_mesh = results
                shape_txt = scene_editing_shape[shape_mode].format(custom_mesh=custom_mesh)
            elif len(results) == 5:
                shape_mode,s,x,y,z = results
                shape_txt = scene_editing_shape[shape_mode].format(s=s,x=x,y=y,z=z)
            else:
                raise NotImplementedError(shape_mode)


        # Use regex to assign our bsdf name to all shapes
        if brdf_mode["name"] != 'gt':
            assert brdf_mode["name"] in scene_editing_bsdf
            shape_txt = re.sub('ref id=".*"', 'ref id="my-bsdf" name="bsdf"', shape_txt)
            scene_txt = re.sub(r'<include filename="materials_(.*).xml"/>', r'<include filename="materials_\1.xml"/>' + f"{scene_editing_bsdf[brdf_mode['name']]}\n".format(**brdf_mode["bsdf_kwargs"]) , scene_txt)

        if emitter_mode != 'same':
            emitter_mode,flashlight_energy = emitter_mode.split("/")
            assert emitter_mode in scene_editing_emitter
            emitters_txt = scene_editing_emitter[emitter_mode].format(intensity=flashlight_energy)

        # Insert our bsdf to the list of BSDFs
        scene_txt = re.sub('<include filename="shapes.xml"/>', '<include filename="shapes.modified.xml"/>', scene_txt)

        scene_txt = re.sub('<include filename="emitters.xml"/>', '<include filename="emitters.modified.xml"/>', scene_txt)
        modified_scene_file = file.parent / (file.stem + ".modified.xml")
        modified_shape_file = shapes_file.parent / (shapes_file.stem + ".modified.xml")
        modified_emitter_file = emitters_file.parent / (emitters_file.stem + ".modified.xml")

        modified_scene_file.write_text(scene_txt, "utf-8")
        modified_shape_file.write_text(shape_txt, "utf-8")
        modified_emitter_file.write_text(emitters_txt, "utf-8")

        # with tempfile.TemporaryDirectory() as tmpdir:
        #     modified_scene_file = Path(tmpdir) / (file.stem + ".modified.xml")
        #     modified_shape_file = Path(tmpdir) / (shapes_file.stem + ".modified.xml")
        #     modified_emitter_file = Path(tmpdir)  / (emitters_file.stem + ".modified.xml")

        #     modified_scene_file.write_text(scene_txt, "utf-8")
        #     modified_shape_file.write_text(shape_txt, "utf-8")
        #     modified_emitter_file.write_text(emitters_txt, "utf-8")

        scene = mi.load_file(str(modified_scene_file))


    else:
        raise Exception('Obsolete code, update the scene!')
        file = Path(file)
        data = file.read_text("utf-8")

        # Use regex to assign our bsdf name to all shapes
        modified_data = re.sub('ref id=".*"', 'ref id="my-bsdf"', data)
        # Insert our bsdf to the list of BSDFs
        modified_data = modified_data.replace("<shape", f"{scene_editing_bsdf[brdf_mode]}\n<shape", 1)

        modified_file = file.parent / (file.stem + ".modified.xml")
        modified_file.write_text(modified_data, "utf-8")
        scene = mi.load_file(str(modified_file))
        modified_file.unlink()
    return scene


builtin_bsdf_required_textures = {
    "constdiffbsdf":[],
    "diffuse": ["reflectance"],
    "principled": ["base_color", "roughness"],
    "principled_diffuse": ["base_color"],
    "principledmy": ["base_color", "roughness", "eta"],
}
#<boolean name="flip_normals" value="true" />
scene_editing_shape = {
    # we need dummy shape to make things work
    'dummy_sdf':
    """
    <scene version="2.1.0">
    <shape type="sphere" id="dummy_sdf_">
      <transform name="to_world">
        <scale value="{s}"/>
	<translate z="{z}"/>
	<translate y="{y}"/>
	<translate x="{x}"/>
      </transform>
      <ref id="my-bsdf" name="bsdf" />
    </shape>
    <shape type="sphere" id="dummy_grad_">
      <transform name="to_world">
        <scale value="{s}"/>
	<translate z="{z}"/>
	<translate y="{y}"/>
	<translate x="{x}"/>
      </transform>
     <bsdf type="diffuse" id="bsdf_grad_activator_">
       <rgb value="0.5 0.5 0.5" name="reflectance" />
     </bsdf>
    </shape>

    </scene>
    """,
    'custom_mesh':
    """
    <scene version="2.1.0">
    <shape type="ply" id="custom_mesh_">
      <string name="filename" value="{custom_mesh}" />
      <ref id="my-bsdf" name="bsdf" />
      <boolean name="face_normals" value="true" />
    </shape>
    </scene>
    """
}
# print("========================temporarily not using face normals for rendering")

scene_editing_emitter = {'flashlight': """<emitter version="3.0.0" id='flashlight' type="point">
                                        <point name="position" value="0.0, 0.0, 0.0"/>
                                        <float name="intensity" value="{intensity}"/>
                                    </emitter>""",
                         'flashlight_sphere': """
                         <shape type="sphere" version="3.0.0" id='flashlight'>
                            <float name="radius" value="0.1"/>
                            <point name="center" x="1" y="0" z="0"/>
                            <emitter type="area"  >
                                  <rgb name="radiance" value="{intensity}"/>

                            </emitter>
                         </shape>""",
                         'spot': """<emitter version="3.0.0" id='flashlight' type="spot">
                         <float name="intensity" value="{intensity}"/>
                         <float name="cutoff_angle" value="90" />
                         <float name="beam_width" value="90" />
                         </emitter>""",
                         'spot_rgb': """<emitter version="3.0.0" id='flashlight' type="spot">
                         <rgb name="intensity" value="{intensity}"/>
                         <float name="cutoff_angle" value="90" />
                         <float name="beam_width" value="90" />
                         </emitter>"""
                         }
#<rgb name="intensity" value="{intensity}"/>
scene_editing_bsdf = {
    "constdiffbsdf": """
    <bsdf type="twosided" id="my-bsdf">
      <bsdf type="diffuse">
        <rgb name="reflectance" value="{color}" />
      </bsdf>
    </bsdf>
    """,
    "mydiffbsdf": """
    <bsdf type="twosided" id="my-bsdf">
        <bsdf type="mydiffbsdf">
        </bsdf>
    </bsdf>""",
    "diffuse": """
     <bsdf type="twosided" id="my-bsdf">
        <bsdf type="diffuse">
            <texture name="reflectance" type="mytexture">
            </texture>
        </bsdf>
    </bsdf>""",
    "principled": """
    <bsdf type="twosided" id="my-bsdf">
        <bsdf type="principled">
            <texture name="base_color" type="mytexture">
            </texture>
            <texture name="roughness" type="mytexture">
            </texture>
            <float name="metallic" value="$metallic" />
            <float name="specular" value="$specular" />
            <float name="spec_tint" value="$spec_tint" />
            <float name="anisotropic" value="$anisotropic" />
            <float name="sheen" value="$sheen" />
            <float name="sheen_tint" value="$sheen_tint" />
            <float name="clearcoat" value="$clearcoat" />
            <float name="clearcoat_gloss" value="$clearcoat_gloss" />
            <float name="spec_trans" value="$spec_trans" />
        </bsdf>
    </bsdf>""",
    "principled_diffuse": """
    <bsdf type="twosided" id="my-bsdf">
        <bsdf type="principled">
            <texture name="base_color" type="mytexture">
            </texture>
            <float name="roughness" value="1.0" />
            <float name="metallic" value="$metallic" />
            <float name="specular" value="$specular" />
            <float name="spec_tint" value="$spec_tint" />
            <float name="anisotropic" value="$anisotropic" />
            <float name="sheen" value="$sheen" />
            <float name="sheen_tint" value="$sheen_tint" />
            <float name="clearcoat" value="$clearcoat" />
            <float name="clearcoat_gloss" value="$clearcoat_gloss" />
            <float name="spec_trans" value="$spec_trans" />
        </bsdf>
    </bsdf>""",
    "principledmy": """
    <bsdf type="twosided" id="my-bsdf">
        <bsdf type="principledmy">
            <texture name="base_color" type="mytexture">
            </texture>
            <texture name="roughness" type="mytexture">
            </texture>
            <texture name="eta" type="mytexture">
            </texture>
            <float name="metallic" value="$metallic" />
            <float name="spec_tint" value="$spec_tint" />
            <float name="anisotropic" value="$anisotropic" />
            <float name="sheen" value="$sheen" />
            <float name="sheen_tint" value="$sheen_tint" />
            <float name="clearcoat" value="$clearcoat" />
            <float name="clearcoat_gloss" value="$clearcoat_gloss" />
            <float name="spec_trans" value="$spec_trans" />
        </bsdf>
    </bsdf>"""
    ,
}

            # <texture name="clearcoat" type="mytexture">
            # </texture>
            # <texture name="clearcoat_gloss" type="mytexture">
            # </texture>
# """
#     <bsdf type="twosided" id="my-bsdf">
#         <bsdf type="principled">
#             <float name="base_color" value="0.5" />

#             <float name="roughness" value="0.5" />

#             <float name="metallic" value="$metallic" />
#             <float name="specular" value="$specular" />
#             <float name="spec_tint" value="$spec_tint" />
#             <float name="anisotropic" value="$anisotropic" />
#             <float name="sheen" value="$sheen" />
#             <float name="sheen_tint" value="$sheen_tint" />
#             <float name="clearcoat" value="$clearcoat" />
#             <float name="clearcoat_gloss" value="$clearcoat_gloss" />
#             <float name="spec_trans" value="$spec_trans" />
#         </bsdf>
#     </bsdf>""",

def load_scene_with_roughness_data(file) -> mi.Scene:
    """Edit XML and assign one single SVBRDF to all shapes in the scene"""

    if 'custom_kitch' in file or 'cornell-box-nobox' in file or 'cornell-box' or 'living-room-2' in file or 'veach_ajar' in file or 'nerf_scenes' in file or 'bunny' in file:
        file = Path(file)
        scene_txt = (file).read_text("utf-8")

        brdf_file = file.parent/'materials_principled.xml'
        brdf_txt = (brdf_file).read_text("utf-8")

        # Replace all roughness with everything
        brdf_txt = brdf_txt.replace('base_color','halalalaunqiue')
        brdf_txt = brdf_txt.replace('roughness','base_color')
        brdf_txt = brdf_txt.replace('halalalaunqiue','roughness')

        scene_txt = re.sub('<include filename="materials_principled.xml"/>', '<include filename="materials_principled.modified_roughness.xml"/>', scene_txt)

        modified_scene_file = file.parent / (file.stem + ".modified_roughness.xml")
        modified_brdf_file = brdf_file.parent / (brdf_file.stem + ".modified_roughness.xml")

        modified_scene_file.write_text(scene_txt, "utf-8")
        modified_brdf_file.write_text(brdf_txt, "utf-8")

        scene = mi.load_file(str(modified_scene_file))
        modified_scene_file.unlink()
        modified_brdf_file.unlink()

    else:
        raise Exception('Obsolete code, update the scene!')

        #left here for backward compatibility
        file = Path(file)
        data = file.read_text("utf-8")

        # Replace all roughness with everything
        data = data.replace('base_color','halalalaunqiue')
        data = data.replace('roughness','base_color')
        data = data.replace('halalalaunqiue','roughness')

        modified_file = file.parent / (file.stem + ".modified_roughness.xml")
        modified_file.write_text(data, "utf-8")
        scene = mi.load_file(str(modified_file))
        modified_file.unlink()
    return scene



def swap_roughness_net_and_albedo_net(params, using_gt_brdf = False, scene_file = None):
    if not using_gt_brdf:
        reflectance_name = 'my-bsdf.brdf_0.base_color.texture'
        reflectance = params[reflectance_name]
        ref_params = mi.traverse(reflectance)

        # find learned (mitsuba_wrapper) key
        key = [k for k in ["network", "tensor", "mi_texture"] if k in ref_params][0]
        ref_net = ref_params[key]

        roughness_name = 'my-bsdf.brdf_0.roughness.texture'
        roughness = params[roughness_name]
        rough_params = mi.traverse(roughness)
        rough_net = rough_params[key]

        match key:
            case "network":
                roughness.network.network = ref_net
                reflectance.network.network = rough_net
            case "tensor":
                roughness.network.tensor = ref_net
                reflectance.network.tensor = rough_net
            case "mi_texture":
                roughness.network.texture = ref_net
                reflectance.network.texture = rough_net
    else:
        return load_scene_with_roughness_data(scene_file)

def swap_eta_net_and_albedo_net(params, using_gt_brdf = False, scene_file = None):
    if not using_gt_brdf:
        reflectance_name = 'my-bsdf.brdf_0.base_color.texture'
        reflectance = params[reflectance_name]
        ref_params = mi.traverse(reflectance)

        # find learned (mitsuba_wrapper) key
        key = [k for k in ["network", "tensor", "mi_texture"] if k in ref_params][0]
        ref_net = ref_params[key]

        eta_name = 'my-bsdf.brdf_0.eta.texture'
        eta = params[eta_name]
        eta_params = mi.traverse(eta)
        eta_net = eta_params[key]

        match key:
            case "network":
                eta.network.network = ref_net
                reflectance.network.network = eta_net
            case "tensor":
                eta.network.tensor = ref_net
                reflectance.network.tensor = eta_net
            case "mi_texture":
                eta.network.texture = ref_net
                reflectance.network.texture = eta_net
    else:
        raise NotImplementedError()



def get_batch_size(spp):
    """
    Get the maximum power of 2 batch size possible given the 2^30 limit by mitsuba for the wavefront size
    """
    maximum_wavefrontsize= 2**28
    # maximum_wavefrontsize= 2**30
    return 2**int(dr.log2(maximum_wavefrontsize/spp)/2)
