#version 150

uniform int u_choose_tex;
uniform sampler2D u_sampler1;

in vec4 f_color;
in vec4 f_normal;
in vec4 f_light1;
in vec4 f_light2;
in vec4 f_light3;
in vec2 f_texcoord;
in vec3 v_pos;
in vec3 camPos;

out vec4 out_Color;
void main()
{
    vec4 diffuseColor = f_color;//material color
    
    float diffuseTerm1 = clamp(dot(f_normal, f_light1), 0.0, 1.0);
    float diffuseTerm2 = clamp(dot(f_normal, f_light2), 0.0, 1.0);
    float diffuseTerm3 = clamp(dot(f_normal, f_light3), 0.0, 1.0);

	vec3 lightDir1=vec3(f_light1);
	vec3 lightDir2=vec3(f_light2);
	vec3 lightDir3=vec3(f_light3);

	vec3 diff1=vec3(diffuseTerm1);
	vec3 diff2=vec3(diffuseTerm2);
	vec3 diff3=vec3(diffuseTerm3);

	vec3 cameraDir=normalize(camPos-v_pos);

	vec3 ref1=normalize(lightDir1-2.0*vec3(f_normal)*dot(lightDir1,vec3(f_normal)));
	vec3 ref2=normalize(lightDir2-2.0*vec3(f_normal)*dot(lightDir2,vec3(f_normal)));
	vec3 ref3=normalize(lightDir3-2.0*vec3(f_normal)*dot(lightDir3,vec3(f_normal)));

	vec3 spec1=vec3(1,1,1)*pow(max(0.0,dot(ref1,cameraDir)),20);
	vec3 spec2=vec3(1,1,1)*pow(max(0.0,dot(ref2,cameraDir)),20);
	vec3 spec3=vec3(1,1,1)*pow(max(0.0,dot(ref3,cameraDir)),20);
	
	vec3 diff=(diff1+diff2+diff3)+0.3;
	vec3 spec=(spec1+spec2+spec3);

    //float totalTerm = (diffuseTerm1 + diffuseTerm2 + diffuseTerm3) / 2 * 0.7 + 0.3;
	//float totalTerm=diffuseTerm1;
	
	vec4 totalTerm=0.5*vec4(diff,1.0)+0.5*vec4(spec,1.0);
	//totalTerm*=1.5;

    if (u_choose_tex != 0)
        out_Color = texture( u_sampler1, vec2(f_texcoord)) * totalTerm;
    else
        out_Color = diffuseColor * totalTerm;
}
