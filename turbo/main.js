import * as THREE from 'three';

import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { TransformControls } from 'three/examples/jsm/controls/TransformControls.js';
import { io } from "https://cdn.socket.io/4.7.2/socket.io.esm.min.js";

let cameraPersp, cameraPerspTransform, currentCamera, cameraPerspTransformHelper;
let scene, renderer, control, orbit;

let roomid=new Date().getTime();

let cposes=[];
let cmatrix=[];

var socket = io();
socket.on('connect', function() {
  socket.emit('server_reconnect', {roomid: roomid});
});


socket.on("server_response", function (msg,ack) {
    console.log(msg);
    //接收到后端发送过来的消息
    var b64img = msg.b64img;
    if(!b64img)return;
    const imageContainer = document.querySelector('.imageContainer');

    var image = new Image();
    image.src = 'data:image/jpeg;base64,' + b64img;

    while (imageContainer.firstChild) {
        imageContainer.removeChild(imageContainer.firstChild);
    }

    imageContainer.prepend(image);
  });

init();
render();



function detect_change(){
    var matrix=cameraPerspTransform.matrix.elements;
    matrix[3]=cameraPerspTransform.position.x;
    matrix[7]=cameraPerspTransform.position.y;
    matrix[11]=cameraPerspTransform.position.z;
    matrix=matrix.slice(0, 12);

    if(JSON.stringify(cmatrix)!=JSON.stringify(matrix)){
        cposes.push(matrix);
        cmatrix=matrix;
    }else{
        if(cposes.length){
            console.log(cposes);
            socket.emit('camera_poses', {roomid:roomid,camera_poses:JSON.stringify(cposes)});
        }

        cposes=[];
    }

    setTimeout(function(){
        detect_change();
    },100);
}

function init() {

    renderer = new THREE.WebGLRenderer( { antialias: true } );
    renderer.setPixelRatio( window.devicePixelRatio );
    renderer.setSize( window.innerWidth, window.innerHeight );
    document.body.appendChild( renderer.domElement );

    const aspect = window.innerWidth / window.innerHeight;

    cameraPersp = new THREE.PerspectiveCamera( 50, aspect, 0.01, 30000 );
    cameraPerspTransform = new THREE.PerspectiveCamera( 50, aspect, 0.01, 30000 );
    
    cameraPerspTransformHelper = new THREE.CameraHelper( cameraPerspTransform );
                
    currentCamera = cameraPersp;

    currentCamera.position.set( 5, 2.5, 5 );

    scene = new THREE.Scene();
    scene.add( new THREE.GridHelper( 5, 10, 0x888888, 0x444444 ) );

    const ambientLight = new THREE.AmbientLight( 0xffffff );
    scene.add( ambientLight );

    const light = new THREE.DirectionalLight( 0xffffff, 4 );
    light.position.set( 1, 1, 1 );
    scene.add( light );

    const texture = new THREE.TextureLoader().load( 'textures/crate.gif', render );
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = renderer.capabilities.getMaxAnisotropy();

    const geometry = new THREE.BoxGeometry();
    const material = new THREE.MeshLambertMaterial( { map: texture } );

    orbit = new OrbitControls( currentCamera, renderer.domElement );
    orbit.update();
    orbit.addEventListener( 'change', render );

    control = new TransformControls( currentCamera, renderer.domElement );
    control.addEventListener( 'change', render );

    control.addEventListener( 'dragging-changed', function ( event ) {

        orbit.enabled = ! event.value;

    } );

    const mesh = new THREE.Mesh( geometry, material );
    scene.add( cameraPerspTransform );

    control.attach( cameraPerspTransform );
    scene.add( control );
    scene.add( cameraPerspTransformHelper );
    
    document.getElementById('btn_translate').addEventListener('click',function(){
        control.setMode( 'translate' );
    });
    
    document.getElementById('btn_rotate').addEventListener('click',function(){
        control.setMode( 'rotate' );
    });
    
    document.getElementById('btn_addpoint').addEventListener('click',function(){
        var ret=JSON.parse(document.getElementById('tb_result').value);
        var matrix=cameraPerspTransform.matrix.elements;
        matrix[3]=cameraPerspTransform.position.x;
        matrix[7]=cameraPerspTransform.position.y;
        matrix[11]=cameraPerspTransform.position.z;
        matrix=matrix.slice(0, 12);
        ret.push(matrix);
        document.getElementById('tb_result').value=JSON.stringify(ret);
    });

    document.getElementById('btn_startrt').addEventListener('click',function(){
        detect_change();
    });

    window.addEventListener( 'resize', onWindowResize );

    window.addEventListener( 'keydown', function ( event ) {

        switch ( event.keyCode ) {

            case 81: // Q
                control.setSpace( control.space === 'local' ? 'world' : 'local' );
                break;

            case 16: // Shift
                control.setTranslationSnap( 100 );
                control.setRotationSnap( THREE.MathUtils.degToRad( 15 ) );
                control.setScaleSnap( 0.25 );
                break;

            case 87: // W
                control.setMode( 'translate' );
                break;

            case 69: // E
                control.setMode( 'rotate' );
                break;

            case 82: // R
                control.setMode( 'scale' );
                break;

            case 67: // C
                const position = currentCamera.position.clone();

                currentCamera = currentCamera.isPerspectiveCamera ? cameraPerspTransform : cameraPersp;
                currentCamera.position.copy( position );

                orbit.object = currentCamera;
                control.camera = currentCamera;

                currentCamera.lookAt( orbit.target.x, orbit.target.y, orbit.target.z );
                onWindowResize();
                break;

            case 86: // V
                const randomFoV = Math.random() + 0.1;
                const randomZoom = Math.random() + 0.1;

                cameraPersp.fov = randomFoV * 160;
                cameraPerspTransform.bottom = - randomFoV * 500;
                cameraPerspTransform.top = randomFoV * 500;

                cameraPersp.zoom = randomZoom * 5;
                cameraPerspTransform.zoom = randomZoom * 5;
                onWindowResize();
                break;

            case 187:
            case 107: // +, =, num+
                control.setSize( control.size + 0.1 );
                break;

            case 189:
            case 109: // -, _, num-
                control.setSize( Math.max( control.size - 0.1, 0.1 ) );
                break;

            case 88: // X
                control.showX = ! control.showX;
                break;

            case 89: // Y
                control.showY = ! control.showY;
                break;

            case 90: // Z
                control.showZ = ! control.showZ;
                break;

            case 32: // Spacebar
                control.enabled = ! control.enabled;
                break;

            case 27: // Esc
                control.reset();
                break;

        }

    } );

    window.addEventListener( 'keyup', function ( event ) {

        switch ( event.keyCode ) {

            case 16: // Shift
                control.setTranslationSnap( null );
                control.setRotationSnap( null );
                control.setScaleSnap( null );
                break;

        }

    } );

}

function onWindowResize() {

    const aspect = window.innerWidth / window.innerHeight;

    cameraPersp.aspect = aspect;
    cameraPersp.updateProjectionMatrix();

    cameraPerspTransform.left = cameraPerspTransform.bottom * aspect;
    cameraPerspTransform.right = cameraPerspTransform.top * aspect;
    cameraPerspTransform.updateProjectionMatrix();

    renderer.setSize( window.innerWidth, window.innerHeight );

    render();

}

function render() {

    renderer.render( scene, currentCamera );

}
