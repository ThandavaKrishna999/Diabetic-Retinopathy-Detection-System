// 3D Background Scene using Three.js v2.0
// Features: Neural Network Constellation Effect & Cyber Eye Geometry

document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('canvas-container');
    if (!container) return;

    // Scene Setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    // --- 1. Neural Network Constellation (Particles + Lines) ---
    const particlesGeometry = new THREE.BufferGeometry();
    const particlesCount = 350; // Reduced count for line performance
    
    const posArray = new Float32Array(particlesCount * 3);
    
    for(let i = 0; i < particlesCount * 3; i++) {
        posArray[i] = (Math.random() - 0.5) * 20; // Spread wider
    }
    
    particlesGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
    
    const particlesMaterial = new THREE.PointsMaterial({
        size: 0.05,
        color: 0x6EE7F9, // Soft Teal
        transparent: true,
        opacity: 0.7,
    });
    
    const particlesMesh = new THREE.Points(particlesGeometry, particlesMaterial);
    scene.add(particlesMesh);

    // Lines connecting particles
    const lineMaterial = new THREE.LineBasicMaterial({
        color: 0x6EE7F9,
        transparent: true,
        opacity: 0.12
    });

    // --- 2. Cyber Eye (Central Geometry) ---
    const eyeGroup = new THREE.Group();
    
    // Core Sphere
    const coreGeo = new THREE.IcosahedronGeometry(1.2, 2);
    const coreMat = new THREE.MeshBasicMaterial({ 
        color: 0x80D0FF, 
        wireframe: true,
        transparent: true, 
        opacity: 0.25 
    });
    const coreMesh = new THREE.Mesh(coreGeo, coreMat);
    eyeGroup.add(coreMesh);
    
    // Outer Ring
    const ringGeo = new THREE.TorusGeometry(2.2, 0.02, 16, 100);
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x6EE7F9 });
    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
    eyeGroup.add(ringMesh);
    
    // Inner Ring (Rotating opposite)
    const ring2Geo = new THREE.TorusGeometry(1.8, 0.03, 16, 100);
    const ring2Mat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.5 });
    const ring2Mesh = new THREE.Mesh(ring2Geo, ring2Mat);
    eyeGroup.add(ring2Mesh);

    eyeGroup.position.set(4, 0, -5); // Position to the right
    scene.add(eyeGroup);

    camera.position.z = 5;

    // Mouse Interaction
    let mouseX = 0;
    let mouseY = 0;
    
    document.addEventListener('mousemove', (event) => {
        mouseX = (event.clientX / window.innerWidth) * 2 - 1;
        mouseY = -(event.clientY / window.innerHeight) * 2 + 1;
    });

    // Animation Loop
    const clock = new THREE.Clock();

    const tick = () => {
        const elapsedTime = clock.getElapsedTime();

        // Rotate Particle Field
        particlesMesh.rotation.y = elapsedTime * 0.02;
        particlesMesh.rotation.x = mouseY * 0.05;

        // Rotate Eye Group
        eyeGroup.rotation.y = elapsedTime * 0.2;
        eyeGroup.rotation.z = elapsedTime * 0.05;
        coreMesh.rotation.x = elapsedTime * 0.5;
        ringMesh.rotation.x = Math.PI / 2 + Math.sin(elapsedTime * 0.5) * 0.2;
        ring2Mesh.rotation.x = Math.PI / 2 + Math.cos(elapsedTime * 0.5) * 0.2;

        // Dynamic Lines (Neural Effect)
        // Note: Creating lines every frame is expensive, doing simple rotation instead for web performance
        // To simulate neural connections efficiently without killing CPU:
        // We rely on the density of particles rotating to create overlapping visual patterns.

        // Interactive Parallax
        camera.position.x += (mouseX * 0.5 - camera.position.x) * 0.05;
        camera.position.y += (mouseY * 0.5 - camera.position.y) * 0.05;
        camera.lookAt(scene.position);

        renderer.render(scene, camera);
        window.requestAnimationFrame(tick);
    }

    tick();

    // Resize Handle
    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
});
