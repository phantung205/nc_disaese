// Custom JavaScript for Diabetes Prediction App

document.addEventListener('DOMContentLoaded', function () {
    // Form validation for manual input
    const manualForm = document.getElementById('manual-form');
    if (manualForm) {
        manualForm.addEventListener('submit', function (e) {
            const inputs = manualForm.querySelectorAll('input[required]');
            let isValid = true;

            inputs.forEach(input => {
                if (!input.value.trim()) {
                    input.classList.add('is-invalid');
                    isValid = false;
                } else {
                    input.classList.remove('is-invalid');
                    input.classList.add('is-valid');
                }
            });

            if (!isValid) {
                e.preventDefault();
                showAlert('Vui lòng điền đầy đủ thông tin!', 'danger');
            }
        });
    }

    // File upload validation
    const fileInput = document.querySelector('#file');
    if (fileInput) {
        fileInput.addEventListener('change', function () {
            const file = this.files[0];
            if (file) {
                const allowedExtensions = ['.csv', '.xlsx', '.xls'];
                const fileName = file.name.toLowerCase();
                const isValidType = allowedExtensions.some(ext => fileName.endsWith(ext));

                if (!isValidType) {
                    showAlert('Chỉ chấp nhận file CSV hoặc Excel!', 'warning');
                    this.value = '';
                } else if (file.size > 10 * 1024 * 1024) { // 10MB limit
                    showAlert('File không được vượt quá 10MB!', 'warning');
                    this.value = '';
                }
            }
        });
    }

    // Auto-hide alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });
});

// Utility function to show alerts
function showAlert(message, type) {
    const alertContainer = document.querySelector('.container-fluid') || document.querySelector('.container');
    const alertHTML = `
        <div class="alert alert-${type} alert-dismissible fade show mt-3" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    alertContainer.insertAdjacentHTML('afterbegin', alertHTML);
}