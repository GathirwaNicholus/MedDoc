// Select form elements
const form = document.querySelector('form');
const nameInput = document.getElementById('name');
const emailInput = document.getElementById('email'); 
const messageInput = document.getElementById('message');

// Handle form submit
form.addEventListener('submit', e => {

  e.preventDefault();

  // Validate fields
  if(!validateName(nameInput.value)) {
    alert('Please enter your name');
    return;
  }

  if(!validateEmail(emailInput.value)) {
    alert('Please enter a valid email');
    return; 
  }

  // Send form data to server 
  sendFormData(nameInput.value, emailInput.value, messageInput.value);

});

// Validate name field
function validateName(name) {
  return name.length > 0;
}

// Validate email field 
function validateEmail(email) {
  // Regex to validate email
  const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return re.test(email);
}

// Send form data 
function sendFormData(name, email, message) {

  // AJAX request to submit form
  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/contact'); 
  xhr.setRequestHeader('Content-Type', 'application/json');
  xhr.onload = function() {
    console.log('Form submitted!');
  };
  xhr.send(JSON.stringify({
    name,
    email, 
    message
  }));

}
