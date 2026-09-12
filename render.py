"""VMAX v2: OpenGL motorsport scene with analytical v1 contact truth."""
import os, json, math, pathlib, subprocess, argparse
import numpy as np
import moderngl
from PIL import Image, ImageDraw, ImageFont
import geometry as g

ROOT=pathlib.Path(__file__).parent; OUT=ROOT/'output'; OUT.mkdir(exist_ok=True)
WIDTH,HEIGHT=1280,720
g.W,g.H=WIDTH,HEIGHT
CAMS=[g.camera('trackside',[18,-7,3.5],[40,15,0],35),
      g.camera('exit',[58,38,4],[38,12,0],42),
      g.camera('broadcast',[60,-18,15],[27,18,0],49)]

VERT='''#version 330
in vec3 in_pos; in vec3 in_normal; in vec3 in_color; in float in_mat; in vec2 in_uv;
uniform mat4 model; uniform mat4 vp; uniform mat4 light_vp;
out vec3 world; out vec3 normal; out vec3 color; out float mat; out vec2 uv; out vec4 shadowpos;
void main(){vec4 w=model*vec4(in_pos,1); world=w.xyz; normal=mat3(model)*in_normal;
color=in_color; mat=in_mat; uv=in_uv; shadowpos=light_vp*w; gl_Position=vp*w;if(in_mat>10.5)gl_Position.z-=.000005*gl_Position.w;}
'''
FRAG='''#version 330
in vec3 world; in vec3 normal; in vec3 color; in float mat; in vec2 uv; in vec4 shadowpos;
uniform vec3 eye; uniform sampler2DShadow shadowmap; uniform sampler2D signage;
out vec4 frag;
float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
void main(){
 vec3 n=normalize(normal); if(!gl_FrontFacing)n=-n;
 vec3 albedo=color; float rough=.45; float metallic=0.;
 if(mat>10.5){rough=.8;}
 else if(mat>9.5){vec4 tex=texture(signage,uv); albedo=tex.rgb; rough=.7;}
 else if(mat>8.5){frag=vec4(color,1);return;}
 else if(mat>7.5){albedo*=.45;rough=.08;metallic=.65;}
 else if(mat>6.5){rough=.8;}
 else if(mat>5.5){metallic=.7;rough=.25;}
 else if(mat>4.5){float grain=hash(floor(world.xy*150));albedo*=.90+.16*grain;rough=1.;}
 else if(mat>3.5){float grain=hash(floor(world.xy*55));albedo*=.90+.18*grain;rough=1.;}
 else if(mat>2.5){float grain=hash(floor(world.xy*220));albedo*=.965+.07*grain;rough=.95;
   float rad=length(world.xy-vec2(0,40));float racing=exp(-pow((rad-42.)/1.1,2.));albedo*=1.-.18*racing;
 }
 else if(mat>1.5){float weave=step(.5,fract(world.x*220))*step(.5,fract(world.y*220));albedo*=.75+.25*weave;rough=.4;}
 else if(mat>.5){rough=.8;albedo*=.85+.12*hash(world.xz*80.);}
 else{rough=.22;metallic=.28;}
 vec3 l=normalize(vec3(-.55,-.35,1.)); vec3 v=normalize(eye-world); vec3 h=normalize(v+l);
 float ndl=max(dot(n,l),0.); float spec=pow(max(dot(n,h),0.),mix(110.,8.,rough));
 vec3 sp=shadowpos.xyz/shadowpos.w*.5+.5; float visibility=1.;
 if(all(greaterThan(sp,vec3(0)))&&all(lessThan(sp,vec3(1)))){
   visibility=0.; float bias=max(.0012*(1.-ndl),.00065);
   for(int x=-1;x<=1;x++)for(int y=-1;y<=1;y++)visibility+=texture(shadowmap,vec3(sp.xy+vec2(x,y)/2048.,sp.z-bias));
   visibility/=9.;
 }
 float sky=.34+.12*max(n.z,0.); vec3 light=albedo*(sky+visibility*ndl*.85);
 light+=vec3(1.,.94,.82)*spec*(.15+.6*metallic)*visibility;
 light+=albedo*.07*max(dot(n,normalize(vec3(.4,.4,.5))),0.);
 float fog=1.-exp(-length(eye-world)*.0008); light=mix(light,vec3(.63,.74,.8),fog);
 light=pow(max(light,vec3(0)),vec3(.82));frag=vec4(light,1);
}'''
SHADOW_VERT='''#version 330
in vec3 in_pos; uniform mat4 model; uniform mat4 light_vp;
void main(){gl_Position=light_vp*model*vec4(in_pos,1);}'''
SHADOW_FRAG='''#version 330
void main(){}'''

class Mesh:
    def __init__(self): self.parts=[]
    def tri(self,p,color,mat=0,normals=None,uv=None):
        p=np.array(p,float)
        if normals is None:
            n=np.cross(p[1]-p[0],p[2]-p[0]); n/=max(np.linalg.norm(n),1e-15); normals=np.tile(n,(3,1))
        self.parts.append(np.column_stack([p,normals,np.tile(np.array(color)/255,(3,1)),np.full(3,mat),np.zeros((3,2)) if uv is None else uv]))
    def quad(self,p,color,mat=0):
        p=np.array(p)
        self.tri(p[[0,1,2]],color,mat,uv=[[0,0],[1,0],[1,1]])
        self.tri(p[[0,2,3]],color,mat,uv=[[0,0],[1,1],[0,1]])
    def box(self,c,size,color,mat=0):
        for p,col in g.box(c,size,color): self.tri(p,col,mat)
    def rod(self,a,b,r,color,mat=0,segments=12):
        a=np.array(a,float); b=np.array(b,float); axis=b-a; axis/=np.linalg.norm(axis)
        u=np.cross(axis,[0,0,1] if abs(axis[2])<.9 else [0,1,0]);u/=np.linalg.norm(u);v=np.cross(axis,u)
        radial=np.array([np.cos(t)*u+np.sin(t)*v for t in np.arange(segments)*2*np.pi/segments])
        p=a+r*radial; q=b+r*radial
        for i in range(segments):
            j=(i+1)%segments
            self.tri([p[i],q[i],q[j]],color,mat,[radial[i],radial[i],radial[j]])
            self.tri([p[i],q[j],p[j]],color,mat,[radial[i],radial[j],radial[j]])
            self.tri([a,p[j],p[i]],color,mat); self.tri([b,q[i],q[j]],color,mat)
    def ellipsoid(self,c,size,color,mat=0):
        c=np.array(c); size=np.array(size)
        for i in range(12):
            for j in range(24):
                points=[]; normals=[]
                for a,b in [(i,j),(i+1,j),(i+1,j+1),(i,j+1)]:
                    phi=a*np.pi/12;theta=b*2*np.pi/24
                    unit=np.array([np.sin(phi)*np.cos(theta),np.sin(phi)*np.sin(theta),np.cos(phi)])
                    points.append(c+size*unit); norm=unit/size; normals.append(norm/np.linalg.norm(norm))
                for k in [[0,1,2],[0,2,3]]:self.tri(np.array(points)[k],color,mat,np.array(normals)[k])
    def loft(self,sections,color,mat=0,sides=24):
        # Sections: x, lateral centre, height centre, lateral radius, height radius.
        rings=[]
        for x,y,z,ry,rz in sections:
            rings.append(np.array([[x,y+ry*np.cos(t),z+rz*np.sin(t)] for t in np.arange(sides)*2*np.pi/sides]))
        for i in range(len(rings)-1):
            for j in range(sides):
                k=(j+1)%sides; pts=np.array([rings[i][j],rings[i+1][j],rings[i+1][k],rings[i][k]])
                norms=[]
                for sec,idx in [(i,j),(i+1,j),(i+1,k),(i,k)]:
                    angle=idx*2*np.pi/sides;n=np.array([0,np.cos(angle)/sections[sec][3],np.sin(angle)/sections[sec][4]]);norms.append(n/np.linalg.norm(n))
                for ids in [[0,1,2],[0,2,3]]:self.tri(pts[ids],color,mat,np.array(norms)[ids])
        for ring in [rings[0],rings[-1]]:
            for j in range(sides): self.tri([ring.mean(0),ring[j],ring[(j+1)%sides]],color,mat)
    def tube(self,points,r,color,mat=0):
        for a,b in zip(points[:-1],points[1:]):self.rod(a,b,r,color,mat,10)
    def array(self): return np.concatenate(self.parts).astype('f4')

CARBON=(24,28,33); RUBBER=(22,24,27); SILVER=(110,118,125); WHITE=(230,235,231)
def body(color):
    m=Mesh()
    # Low floor with a tapered boat tail and sidepod undercuts.
    m.loft([(-2.45,0,.18,.2,.08),(-1.6,0,.17,.68,.08),(.6,0,.17,.76,.08),(1.4,0,.18,.35,.07)],CARBON,2)
    m.loft([(-2.2,0,.4,.18,.17),(-1.2,0,.52,.44,.3),(-.45,0,.56,.44,.32),(.5,0,.5,.35,.25),(1.1,0,.44,.25,.19),(2.55,0,.27,.12,.065)],color)
    for side in [-1,1]:
        m.loft([(-1.85,side*.38,.36,.12,.12),(-1.3,side*.53,.37,.24,.18),(-.6,side*.57,.43,.29,.25),(.15,side*.56,.47,.27,.24),(.65,side*.5,.48,.2,.16)],color)
        m.box((.64,side*.51,.48),(.015,.29,.20),CARBON,2) # sidepod cooling inlet
        m.box((-.2,side*.82,.18),(1.65,.06,.08),CARBON,2)
        for x in np.arange(-1.4,-.4,.11):m.box((x,side*.60,.63),(.035,.25,.017),CARBON,2)
    # Cockpit recess, driver helmet and air intake above engine cover.
    m.ellipsoid((-.13,0,.77),(.51,.29,.10),CARBON,2)
    m.ellipsoid((-.23,0,.86),(.18,.16,.20),(245,211,35))
    m.ellipsoid((-.10,0,.88),(.075,.155,.07),(35,52,70),8)
    m.loft([(-2.0,0,.53,.06,.12),(-1.2,0,.72,.14,.31),(-.72,0,.9,.17,.29)],color)
    m.ellipsoid((-.70,0,1.04),(.04,.125,.115),CARBON,2)
    # Halo is a continuous curved load-bearing ring with a front support.
    a=np.linspace(-np.pi/2,np.pi/2,28)
    halo=np.column_stack([.02+.64*np.cos(a),.35*np.sin(a),np.full(len(a),1.00)])
    m.tube(halo,.035,CARBON,2);m.rod([.66,0,1],[.83,0,.53],.042,CARBON,2)
    for side in [-1,1]:
        m.rod([.02,side*.35,1],[-.58,side*.34,.78],.035,CARBON,2)
        m.rod([.47,side*.35,.67],[.38,side*.68,.66],.018,CARBON,2)
        m.ellipsoid((.38,side*.69,.67),(.12,.06,.05),color)
    # Multi-element front wing, flap curvature and endplates.
    for index in range(4):
        x=2.64-index*.16;z=.14+index*.055
        m.loft([(x-.11,0,z,.99,.014),(x,0,z+.025,.99,.018),(x+.07,0,z,.98,.012)],color if index==2 else CARBON,2 if index!=2 else 0)
    for side in [-1,1]:m.box((2.4,side*.98,.26),(.64,.04,.29),color)
    # Curved rear wing and twin support pylons, with rear diffuser strakes.
    for x,z in [(-2.30,.86),(-2.54,1.00)]:
        m.loft([(x-.12,0,z,.91,.018),(x,0,z-.055,.91,.025),(x+.13,0,z+.005,.91,.016)],color)
    for side in [-1,1]:
        m.box((-2.43,side*.91,.85),(.60,.045,.46),color)
        m.rod([-1.95,side*.2,.36],[-2.3,side*.2,.86],.035,CARBON,2)
    for y in np.linspace(-.55,.55,7):m.box((-2.08,y,.21),(.7,.022,.18),CARBON,2)
    m.box((-2.44,0,.32),(.05,.13,.075),(255,22,13),9)
    # Wishbones connect to the fixed ideal tyre centres.
    for x,y in g.CONTACT:
        for z in [.23,.43]:
            for dx in [-.42,.32]:m.rod([x+dx,0,z],[x,y,.36],.026,CARBON,2)
        m.rod([x+.18,.2*np.sign(y),.62],[x,y,.36],.023,CARBON,2)
    # Original VMAX livery panels, applied as geometry-aligned textured decals.
    for side in [-1,1]:
        y=side*.845
        m.quad([[-1.13,y,.57],[.02,y,.57],[.02,y,.41],[-1.13,y,.41]],WHITE,10)
    m.quad([[-2.68,-.65,1.035],[-2.68,.65,1.035],[-2.68,.65,.935],[-2.68,-.65,.935]],WHITE,10)
    # Contrast stripe along the nose.
    m.loft([(.85,0,.63,.07,.008),(1.25,0,.59,.055,.008),(2.4,0,.338,.035,.008)],WHITE,7)
    return m

def wheel(side):
    m=Mesh(); axis=np.array([0,1,0])
    # Rounded slick tyre profile: extreme radius .36, maximum half-width .18.
    profile=[(-.18,.27),(-.17,.32),(-.13,.352),(-.08,.36),(.08,.36),(.13,.352),(.17,.32),(.18,.27)]
    rings=[]
    for y,r in profile:rings.append(np.array([[r*np.cos(t),y,r*np.sin(t)] for t in np.arange(48)*2*np.pi/48]))
    for i in range(len(rings)-1):
        for j in range(48):
            k=(j+1)%48; p=np.array([rings[i][j],rings[i+1][j],rings[i+1][k],rings[i][k]])
            n=[]
            for row,index in [(i,j),(i+1,j),(i+1,k),(i,k)]:
                dr=profile[min(row+1,len(profile)-1)][1]-profile[max(row-1,0)][1]
                dy=profile[min(row+1,len(profile)-1)][0]-profile[max(row-1,0)][0]
                angle=index*2*np.pi/48; norm=np.array([dy*np.cos(angle),-dr,dy*np.sin(angle)]);n.append(norm/np.linalg.norm(norm))
            for ids in [[0,1,2],[0,2,3]]:m.tri(p[ids],RUBBER,1,np.array(n)[ids])
    for sign in [-1,1]:
        m.rod([0,sign*.172,0],[0,sign*.183,0],.253,(36,39,43),1,40)
        m.rod([0,sign*.18,0],[0,sign*.186,0],.19,(58,63,67),6,40)
        for a in np.arange(10)*2*np.pi/10:
            m.rod([.05*np.cos(a),sign*.19,.05*np.sin(a)],[.175*np.cos(a+.10),sign*.19,.175*np.sin(a+.10)],.013,SILVER,6)
        m.rod([0,sign*.18,0],[0,sign*.205,0],.048,(175,178,182),6,20)
        # Narrow sidewall compound ring and broken lettering-like detail.
        a=np.linspace(0,2*np.pi,65);m.tube(np.column_stack([.29*np.cos(a),np.full(len(a),sign*.176),.29*np.sin(a)]),.006,(246,204,45),7)
        for a in np.arange(16)*2*np.pi/16:
            m.rod([.273*np.cos(a),sign*.182,.273*np.sin(a)],[.283*np.cos(a+.055),sign*.182,.283*np.sin(a+.055)],.006,(232,231,208),7)
    return m

def strip(m,ss,lo,hi,z,col,mat):
    c,n=g.center(ss);a=c+n*lo;b=c+n*hi
    for i in range(len(ss)-1):m.quad([[*a[i],z],[*b[i],z],[*b[i+1],z],[*a[i+1],z]],col,mat)

def scene(ground_only=False):
    m=Mesh();m.quad([[-250,-250,-.08],[250,-250,-.08],[250,250,-.08],[-250,250,-.08]],(77,112,57),4)
    ss=np.linspace(-30,93,750)
    strip(m,ss,-17,17,-.035,(181,167,137),5)
    strip(m,ss,-9.6,9.6,-.012,(43,118,103),7)
    strip(m,ss,-7,7,0,(65,68,72),3)
    for sign in [-1,1]:
        strip(m,ss,min(sign*6.9,sign*7),max(sign*6.9,sign*7),.00001,WHITE,11)
        for i in range(0,len(ss)-1,5):
            strip(m,ss[i:min(i+6,len(ss))],min(sign*7,sign*7.8),max(sign*7,sign*7.8),.003,(208,34,42) if (i//5)%2==0 else WHITE,7)
        if ground_only: continue
        # Concrete crash barrier outside run-off, triple rails and catch fencing.
        c,n=g.center(np.linspace(-30,93,85));p=c+n*(sign*18)
        for a,b in zip(p[:-1],p[1:]):
            m.quad([[*a,0],[*b,0],[*b,1.0],[*a,1.0]],(186,192,188),7)
            for z in [.37,.65,.92]:m.rod([*a,z],[*b,z],.055,(143,153,157),6,6)
        for idx,(a,b) in enumerate(zip(p[:-1],p[1:])):
            m.rod([*a,1.0],[*a,3.6],.035,(117,128,132),6,6)
            for z in [1.4,1.8,2.2,2.6,3.,3.4]:m.rod([*a,z],[*b,z],.009,(97,112,118),6,5)
            for t in [.25,.5,.75]:
                q=a*(1-t)+b*t;m.rod([*q,1.0],[*q,3.6],.007,(97,112,118),6,5)
            if idx%7==0:
                # Textured sponsor boards on the concrete wall.
                m.quad([[*a,1.0],[*b,1.0],[*b,.10],[*a,.10]],WHITE,10)
    if ground_only: return m
    # Grandstand on the infield, stepped seating and a thin canopy.
    rng=np.random.default_rng(27)
    for row in range(7):
        y=25+row*.85;z=.3+row*.48
        m.box((8,y,z),(24,.8,.24),(152,162,164),6)
        for col in range(34):
            x=-3.6+col*.7; m.box((x,y,z+.2),(.48,.5,.22),(44,82,111) if col%3 else (209,215,210),7)
            if rng.random()<.68:
                shirt=[(210,41,41),(230,231,225),(35,57,81),(242,182,35)][int(rng.integers(4))]
                m.box((x,y,z+.55),(.26,.24,.45),shirt,7)
                m.ellipsoid((x,y,z+.85),(.105,.10,.12),(176,142,119),7)
    for x in [-4,4,12,20]:m.rod([x,31.5,0],[x,31.5,6.5],.12,(165,175,176),6)
    m.box((8,28.5,6.5),(26,8,.18),(220,226,222),6)
    # Distant paddock buildings and light poles ground the horizon.
    for x in [-38,-20,0,20]:
        m.box((x,67,3),(15,9,6),(188,198,200),7);m.box((x,62.45,3.5),(13,.05,1.8),(49,76,89),8)
    for p in [[-10,-25],[66,5],[65,65],[-20,45]]:
        m.rod([*p,0],[*p,12],.10,(175,183,185),6);m.box((*p,12),(2,.5,.3),(225,227,221),7)
    return m

def lookat(eye,target,up=np.array([0,0,1.])):
    eye=np.array(eye,float); f=np.array(target)-eye;f/=np.linalg.norm(f);r=np.cross(f,up);r/=np.linalg.norm(r);u=np.cross(r,f)
    view=np.eye(4);view[:3,:3]=[r,u,-f];view[:3,3]=-view[:3,:3]@eye;return view

def vp_camera(cam):
    near,far=.2,500.;f=1/np.tan(np.deg2rad(cam['vertical_fov_deg'])/2)
    projection=np.array([[f/(WIDTH/HEIGHT),0,0,0],[0,f,0,0],[0,0,(far+near)/(near-far),2*far*near/(near-far)],[0,0,-1,0]])
    return projection@lookat(cam['position_m'],cam['look_at_m'])

def model_matrix(pos,heading=0,spin=0):
    c,s=np.cos(heading),np.sin(heading);m=np.eye(4);m[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];m[:3,3]=pos
    if spin:
        c,s=np.cos(spin),np.sin(spin); rot=np.eye(4);rot[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]];m=m@rot
    return m

class Renderer:
    def __init__(self):
        self.ctx=moderngl.create_standalone_context(backend='egl');self.ctx.enable(moderngl.DEPTH_TEST)
        self.prog=self.ctx.program(vertex_shader=VERT,fragment_shader=FRAG)
        self.shadowprog=self.ctx.program(vertex_shader=SHADOW_VERT,fragment_shader=SHADOW_FRAG)
        self.color=self.ctx.texture((WIDTH,HEIGHT),4);self.depth=self.ctx.depth_renderbuffer((WIDTH,HEIGHT))
        self.fbo=self.ctx.framebuffer([self.color],self.depth)
        self.msaa=self.ctx.framebuffer([self.ctx.renderbuffer((WIDTH,HEIGHT),4,samples=4)],self.ctx.depth_renderbuffer((WIDTH,HEIGHT),samples=4))
        self.shadowtex=self.ctx.depth_texture((2048,2048));self.shadowtex.compare_func='<=';self.shadowtex.repeat_x=False;self.shadowtex.repeat_y=False
        self.shadowfbo=self.ctx.framebuffer(depth_attachment=self.shadowtex)
        lightpos=np.array([25,20,0])+np.array([-.55,-.35,1])*100
        lightview=lookat(lightpos,[25,20,0]);extent=95
        ortho=np.diag([1/extent,1/extent,-2/250,1.]);ortho[2,3]=-1
        self.lightvp=ortho@lightview
        self.prog['light_vp'].write(self.lightvp.T.astype('f4').tobytes());self.shadowprog['light_vp'].write(self.lightvp.T.astype('f4').tobytes())
        atlas=Image.new('RGB',(1024,256),(10,24,33));draw=ImageDraw.Draw(atlas)
        font='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
        draw.text((50,8),'VMAX',font=ImageFont.truetype(font,170),fill=(242,247,244))
        draw.polygon([(715,35),(810,35),(720,206),(625,206)],fill=(245,64,45))
        draw.text((825,75),'RACING',font=ImageFont.truetype(font,30),fill=(242,247,244))
        self.sign=self.ctx.texture(atlas.size,3,atlas.tobytes());self.sign.build_mipmaps()
        self.prog['shadowmap']=0;self.prog['signage']=1
        self.scene=self.upload(scene());self.bodies=[self.upload(body((218,27,39))),self.upload(body((11,141,155)))]
        self.wheels=[self.upload(wheel(-1)),self.upload(wheel(1))]
        self.ground=self.upload(scene(ground_only=True));self.backgrounds={}
        self.shadowbase_tex=self.ctx.depth_texture((2048,2048));self.shadowbase_tex.compare_func='<='
        self.shadowbase_tex.repeat_x=False;self.shadowbase_tex.repeat_y=False
        self.shadowbase=self.ctx.framebuffer(depth_attachment=self.shadowbase_tex)
        self.shadowbase.use();self.ctx.viewport=(0,0,2048,2048);self.shadowbase.clear(depth=1)
        self.draw(self.scene,np.eye(4),True)
        self.shadowbytes=self.shadowbase_tex.read()
        print('OpenGL:',self.ctx.info['GL_RENDERER'],flush=True)
    def upload(self,mesh):
        data=mesh.array();buf=self.ctx.buffer(data.tobytes())
        vao=self.ctx.vertex_array(self.prog,[(buf,'3f 3f 3f 1f 2f','in_pos','in_normal','in_color','in_mat','in_uv')])
        svao=self.ctx.vertex_array(self.shadowprog,[(buf,'3f 36x','in_pos')]);return vao,svao,buf
    def draw(self,thing,model,shadow=False):
        prog=self.shadowprog if shadow else self.prog;prog['model'].write(model.T.astype('f4').tobytes());thing[1 if shadow else 0].render()
    def objects(self,rows,travel):
        objects=[(self.scene,np.eye(4))]
        for ci,row in enumerate(rows):
            p=row['world_position'];h=row['heading_rad'];ch=model_matrix([*p,0],h)
            objects.append((self.bodies[ci%2],ch))
            for x,y in g.CONTACT:
                # Kinematic rolling and small front steer do not move the authoritative contacts.
                steer=row.get('visual_steering_rad',0) if x>0 else 0
                local=model_matrix([x,y,.36],steer,-travel[ci]/.36)
                objects.append((self.wheels[int(y>0)],ch@local))
        return objects
    def frame(self,cam,rows,travel):
        objects=self.objects(rows,travel)
        fixed=cam['name'] in [c['name'] for c in CAMS]
        self.prog['vp'].write(vp_camera(cam).T.astype('f4').tobytes());self.prog['eye'].value=tuple(cam['position_m'])
        if fixed and cam['name'] not in self.backgrounds:
            bg=self.ctx.framebuffer([self.ctx.renderbuffer((WIDTH,HEIGHT),4,samples=4)],self.ctx.depth_renderbuffer((WIDTH,HEIGHT),samples=4))
            bg.use();self.ctx.viewport=(0,0,WIDTH,HEIGHT);bg.clear(.58,.71,.79,1,depth=1)
            self.shadowbase_tex.use(0);self.sign.use(1);self.draw(self.scene,np.eye(4))
            self.backgrounds[cam['name']]=bg
        self.shadowtex.write(self.shadowbytes)
        self.shadowfbo.use();self.ctx.viewport=(0,0,2048,2048)
        for obj,model in objects[1:]:self.draw(obj,model,True)
        self.msaa.use();self.ctx.viewport=(0,0,WIDTH,HEIGHT)
        self.shadowtex.use(0);self.sign.use(1)
        if fixed:
            self.ctx.copy_framebuffer(self.msaa,self.backgrounds[cam['name']])
            self.msaa.use();self.ctx.depth_func='<='
            self.draw(self.ground,np.eye(4));self.ctx.depth_func='<'
        else:
            self.msaa.clear(.58,.71,.79,1,depth=1);self.draw(self.scene,np.eye(4))
        for obj,model in objects[1:]:self.draw(obj,model)
        self.ctx.copy_framebuffer(self.fbo,self.msaa)
        return Image.frombytes('RGB',(WIDTH,HEIGHT),self.fbo.read(components=3,alignment=1)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)

def debug(im,cam,rows):
    im=im.copy();d=ImageDraw.Draw(im);ss=np.linspace(-30,93,1600);c,n=g.center(ss)
    for sign in [-1,1]:
        uv,depth=g.project(np.column_stack([c+sign*n*7,np.zeros(len(c))]),cam)
        for i in range(len(uv)-1):
            if min(depth[i:i+2])>.2 and abs(uv[i:i+2]).max()<10000:d.line([tuple(uv[i]),tuple(uv[i+1])],fill=(255,210,40),width=2)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    d.rectangle((20,20,640,65+len(rows)*30),fill=(8,19,28))
    d.text((32,28),'VMAX / CONTACT GEOMETRY',font=font,fill='white')
    for j,row in enumerate(rows):
        p=np.array(list(row['contact_points_world'].values()));uv,depth=g.project(np.column_stack([p,np.zeros(4)]),cam)
        for i,((x,y),z) in enumerate(zip(uv,depth)):
            if z>.2 and 0<=x<WIDTH and 0<=y<HEIGHT:d.ellipse((x-5,y-5,x+5,y+5),fill=(255,45,65) if list(row['corner_excess_m'].values())[i]>0 else (30,255,145))
        d.text((32,57+j*30),f"{row['car_id']} | minimum excess {row['min_excess_m']:+.6f} m",font=font,fill='white')
    return im

def calibrate():
    result={}
    for cam in CAMS:
        s=(np.arange(20000)+.5)/20000*(20*np.pi);c,n=g.center(s);p=np.concatenate([c-n*7,c+n*7]);world=np.column_stack([p,np.zeros(len(p))])
        uv,depth=g.project(world,cam);clip=np.column_stack([world,np.ones(len(world))])@vp_camera(cam).T
        ndc=clip[:,:3]/clip[:,3,None];glpixel=np.column_stack([(ndc[:,0]+1)*WIDTH/2,(1-ndc[:,1])*HEIGHT/2])
        front=depth>.2;error=float(abs(uv[front]-glpixel[front]).max());assert error<1e-7
        hm=np.column_stack([p,np.ones(len(p))])@np.array(cam['ground_plane_homography']).T
        h_error=float(abs(uv[front]-hm[front,:2]/hm[front,2,None]).max());assert h_error<1e-7
        visible=front&(uv[:,0]>=0)&(uv[:,0]<WIDTH)&(uv[:,1]>=0)&(uv[:,1]<HEIGHT)
        cam['boundary_coverage_percent']=float((visible[:20000].mean()*47+visible[20000:].mean()*33)/80*100)
        clip32=np.column_stack([world,np.ones(len(world))]).astype('f4')@vp_camera(cam).T.astype('f4')
        ndc32=clip32[:,:3]/clip32[:,3,None]
        pixel32=np.column_stack([(ndc32[:,0]+1)*WIDTH/2,(1-ndc32[:,1])*HEIGHT/2])
        error32=float(abs(uv[visible]-pixel32[visible]).max());assert error32<.01
        result[cam['name']]={'opengl_vs_pinhole_error_px':error,'homography_vs_pinhole_error_px':h_error,'coverage_percent':cam['boundary_coverage_percent'],'float32_in_frame_error_px':error32}
    (OUT/'calibration.json').write_text(json.dumps(CAMS,indent=2));return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');parser.add_argument('--scenario');args=parser.parse_args()
    report={'cameras':calibrate(),'scenarios':{}};renderer=Renderer()
    sources=sorted((ROOT/'data').glob('*/ground_truth.json'))
    for source in sources:
        name=source.parent.name
        if args.scenario and name!=args.scenario:continue
        if args.preview and name!='side_by_side':continue
        data=json.loads(source.read_text());folder=OUT/name;folder.mkdir(exist_ok=True)
        bycar={cid:[r for r in data['frames'] if r['car_id']==cid] for cid in sorted({r['car_id'] for r in data['frames']})}
        travel=[]
        for rows in bycar.values():
            pos=np.array([r['world_position'] for r in rows]);dist=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pos,axis=0),axis=1))];travel.append(dist)
            headings=np.unwrap([r['heading_rad'] for r in rows]);curvature=np.gradient(headings)/np.maximum(np.gradient(dist),1e-9)
            for r,k in zip(rows,curvature):r['visual_steering_rad']=float(np.clip(np.arctan(3.6*k),-.45,.45))
        for cam in CAMS:
            if args.preview:indices=[48];proc=None
            else:
                indices=range(96)
                proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{WIDTH}x{HEIGHT}','-r','24','-i','-','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/(cam['name']+'.mp4'))],stdin=subprocess.PIPE)
            for i in indices:
                rows=[v[i] for v in bycar.values()];im=renderer.frame(cam,rows,[x[i] for x in travel])
                if proc:proc.stdin.write(im.tobytes())
                if i==48:im.save(folder/(cam['name']+'.png'));debug(im,cam,rows).save(folder/(cam['name']+'_debug.png'))
            if proc:proc.stdin.close();assert proc.wait()==0
            enriched=dict(data);enriched['camera']=cam;enriched['render_version']='VMAX v2';(folder/(cam['name']+'.json')).write_text(json.dumps(enriched,indent=2))
            print(name,cam['name'],'rendered',flush=True)
        report['scenarios'][name]=data['events']
    if not args.preview:(OUT/'verification.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
