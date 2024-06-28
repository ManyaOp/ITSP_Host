$.getScript(
    "https://cdnjs.cloudflare.com/ajax/libs/particles.js/2.0.0/particles.min.js",
    function () {
      function getRandomColor() {
        var letters = "0123456789ABCDEF";
        var color = "#";
        for (var i = 0; i < 6; i++) {
          color += letters[Math.floor(Math.random() * 16)];
        }
        return color;
      }
  
      function updateParticleColors() {
        var particleColor = getRandomColor();
        var lineColor = getRandomColor();
  
        particlesJS("particles-js", {
          particles: {
            number: {
              value: 175,
              density: {
                enable: true,
                value_area: 800,
              },
            },
            color: {
              value: particleColor,
            },
            shape: {
              type: "circle",
              stroke: {
                width: 1,
                color: particleColor,
              },
              polygon: {
                nb_sides: 3,
              },
  
              // "image": {
              //     src:'https://web.iiit.ac.in/~soham.sadavarte/myphoto.jpeg',
              //     "width": 4,
              //     "height": 10
              // }
            },
  
            opacity: {
              value: 1,
              random: false,
              anim: {
                enable: true,
                speed: 0.8,
                opacity_min: 0.1,
                sync: true,
              },
            },
            size: {
              value: 5,
              random: false,
              anim: {
                enable: false,
                speed: 80,
                size_min: 0.5,
                sync: false,
              },
            },
            line_linked: {
              enable: true,
              distance: 150,
              color: lineColor,
              opacity: 0.7,
              width: 2,
            },
            move: {
              enable: true,
              speed: 7,
              direction: "random",
              random: false,
              straight: false,
              out_mode: "out",
              attract: {
                enable: true,
                rotateX: 600,
                rotateY: 1200,
              },
            },
          },
          interactivity: {
            detect_on: "canvas",
            events: {
              onhover: {
                enable: true,
                mode: "repulse",
              },
              onclick: {
                enable: true,
                mode: "push",
              },
              resize: true,
            },
            modes: {
              grab: {
                distance: 200,
                line_linked: {
                  opacity: 1,
                },
              },
  
              repulse: {
                distance: 70,
              },
  
              push: {
                particles_nb: 8,
                size: 10,
              },
              remove: {
                particles_nb: 2,
              },
            },
          },
          retina_detect: true,
        });
  
        setTimeout(updateParticleColors, 5000);
      }
  
      updateParticleColors();
    }
  );
  
  let passwordVisible = false;
  
  function togglePasswordVisibility() {
    const passwordInput = document.getElementById("password");
    const openEyeIcon = document.getElementById("openEye");
    const closedEyeIcon = document.getElementById("closedEye");
  
    passwordVisible = !passwordVisible;
  
    if (passwordVisible) {
      passwordInput.type = "text";
      openEyeIcon.style.display = "inline";
      closedEyeIcon.style.display = "none";
    } else {
      passwordInput.type = "password";
      openEyeIcon.style.display = "none";
      closedEyeIcon.style.display = "inline";
    }
  }

  document.addEventListener("DOMContentLoaded", function() {
    console.log("hi");
    var checkbox = document.getElementById("remember");
    checkbox.checked = localStorage.getItem("remember");
    if (checkbox.checked) {
      console.log("bye");
        var username = localStorage.getItem("username");
        document.getElementById("username").value = username;
        var password = localStorage.getItem("password");
        document.getElementById("password").value = password;
    }
});

function yaad() {
  console.log("chintu");
    var checkbox = document.getElementById("remember");
    localStorage.setItem("remember", checkbox.checked);
    if (checkbox.checked) {
        console.log("priyanka");
        var username = document.getElementById("username").value;
        localStorage.setItem("username", username);
        var password = document.getElementById("password").value;
        localStorage.setItem("password", password);
    } 
    else {
        localStorage.setItem("username", "");
        localStorage.setItem("password", "");
    }
}

