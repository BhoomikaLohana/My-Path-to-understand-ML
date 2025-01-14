#Generating Data
import numpy as np
# Set a seed for reproducibility
np.random.seed(0)

# Generate synthetic data for spam and non-spam emails with a smaller sample size
spam_email_length = np.random.normal(loc=150, scale=5, size=10)
spam_special_chars = np.random.normal(loc=20, scale=2, size=10)

nonspam_email_length = np.random.normal(loc=80, scale=20, size=10)
nonspam_special_chars = np.random.normal(loc=5, scale=2, size=10)

#Plot the Data
import matplotlib.pyplot as plt
plt.scatter(spam_email_length,spam_special_chars, label='Spam Emails')
plt.scatter(nonspam_email_length, nonspam_special_chars, label='nonspam Emails')
plt.xlabel("Email Length")
plt.ylabel(" specail chars")
plt.title("Classification")
plt.legend()
plt.show()

# Prepare data for classification
data_spam = np.vstack((spam_email_length, spam_special_chars)).T
data_nonspam = np.vstack((nonspam_email_length, nonspam_special_chars)).T

# Random linear classifier function
def random_linear_classifier(data_spam, data_nonspam, k, d):
    best_error = float('inf')
    best_theta = None
    best_theta0 = None

    for _ in range(k):
        theta = np.random.normal(size=d)
        theta0 = np.random.normal()

        error = compute_error(data_spam, data_nonspam, theta, theta0)

        if error < best_error:
            best_error = error
            best_theta = theta
            best_theta0 = theta0

    return best_theta, best_theta0

# Compute error function
def compute_error(data_spam, data_nonspam, theta, theta0):
    error = 0
    for x_spam in data_spam:
        if np.dot(theta, x_spam) + theta0 <= 0:
            error += 1
    for x_nonspam in data_nonspam:
        if np.dot(theta, x_nonspam) + theta0 > 0:
            error += 1
    return error

# Apply the random linear classifier
k = 1000 # Number of iterations
d = 2    # Number of dimensions (features)
best_theta, best_theta0 = random_linear_classifier(data_spam, data_nonspam, k, d)

# Plot the data and decision boundary
x_vals = np.linspace(25, 200, 100)
y_vals = (-best_theta[0] / best_theta[1]) * x_vals - (best_theta0 / best_theta[1])

#plot the linear classifier
plt.figure(figsize=(6,4))
plt.scatter(spam_email_length, spam_special_chars, color='red', label='Spam Emails', alpha=0.7)
plt.scatter(nonspam_email_length, nonspam_special_chars, color='green', label='Non-Spam Emails', alpha=0.7)
plt.plot(x_vals, y_vals, color='blue', linestyle='--', label='Decision Boundary')
plt.xlabel("Email Length (words)")
plt.ylabel("Number of Special Characters")
plt.title("Random Linear Classifier - Spam vs Non-Spam Emails")
plt.legend()
plt.grid(True)
plt.show()

from sklearn.model_selection import train_test_split
# Split the spam and non-spam data into training and testing sets (70% for training, 30% for testing)
train_spam, test_spam = train_test_split(data_spam, test_size=0.3, random_state=42)
train_nonspam, test_nonspam = train_test_split(data_nonspam, test_size=0.3, random_state=42)


# Plot the training and testing data points
# Plot training data
plt.scatter(train_spam[:, 0], train_spam[:, 1], color='red', label='Training Spam', alpha=0.7)
plt.scatter(train_nonspam[:, 0], train_nonspam[:, 1], color='green', label='Training Non-Spam', alpha=0.7)

# Plot testing data
plt.scatter(test_spam[:, 0], test_spam[:, 1], color='darkred', label='Testing Spam', alpha=0.7, marker='x')
plt.scatter(test_nonspam[:, 0], test_nonspam[:, 1], color='darkgreen', label='Testing Non-Spam', alpha=0.7, marker='x')

# Labels and title
plt.xlabel("Email Length (words)")
plt.ylabel("Number of Special Characters")
plt.title("Training and Testing Data for Spam vs Non-Spam Emails")
plt.legend()
plt.grid(True)
plt.show()

# Combine training data
X_train = np.vstack((train_spam, train_nonspam))
y_train = np.hstack((np.ones(len(train_spam)), np.zeros(len(train_nonspam))))

# Combine testing data
X_test = np.vstack((test_spam, test_nonspam))
y_test = np.hstack((np.ones(len(test_spam)), np.zeros(len(test_nonspam))))

# Random linear classifier function
def random_linear_classifier(data_spam, data_nonspam, k, d):
    best_error = float('inf')
    best_theta = None
    best_theta0 = None

    for _ in range(k):
        theta = np.random.normal(size=d)
        theta0 = np.random.normal()

        error = compute_error(data_spam, data_nonspam, theta, theta0)

        if error < best_error:
            best_error = error
            best_theta = theta
            best_theta0 = theta0

    return best_theta, best_theta0, best_error

# Compute error function
def compute_error(data_spam, data_nonspam, theta, theta0):
    error = 0
    for x_spam in data_spam:
        if np.dot(theta, x_spam) + theta0 <= 0:
            error += 1
    for x_nonspam in data_nonspam:
        if np.dot(theta, x_nonspam) + theta0 > 0:
            error += 1
    return error

# Parameters
k = 1000
d = 2

# Train the classifier
best_theta_train, best_theta0_train, train_error = random_linear_classifier(
    X_train[y_train == 1], X_train[y_train == 0], k, d
)

# Decision boundary for training data
x_vals_train = np.linspace(X_train[:, 0].min() - 1, X_train[:, 0].max() + 1, 100)
y_vals_train = (-best_theta_train[0] / best_theta_train[1]) * x_vals_train - (best_theta0_train / best_theta_train[1])

# Plot training data with decision boundary
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label="Training Spam", color="red")
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label="Training Non-Spam", color="green")
plt.plot(x_vals_train, y_vals_train, color="blue", linestyle="--", label="Decision Boundary")
plt.xlabel("Email Length (words)")
plt.ylabel("Number of Special Characters")
plt.title("Spam vs Non-Spam Classification (Training Data)")
plt.legend()
plt.show()

# Compute train and test error
test_error = compute_error(X_test[y_test == 1], X_test[y_test == 0], best_theta_train, best_theta0_train)
print(f"Testing error: {test_error}")
print(f"Training error: {train_error}")

# Predict test data labels
predicted_test_labels = np.ones_like(y_test)
for i, x_test in enumerate(X_test):
    if np.dot(best_theta_train, x_test) + best_theta0_train <= 0:
        predicted_test_labels[i] = 0

# Plot test data with predicted labels
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], label="Actual Spam", color="darkred")
plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], label="Actual Non-Spam", color="darkgreen")
plt.scatter(X_test[predicted_test_labels == 1][:, 0], X_test[predicted_test_labels == 1][:, 1], 
            label="Predicted Spam", marker="x", color="orange")
plt.scatter(X_test[predicted_test_labels == 0][:, 0], X_test[predicted_test_labels == 0][:, 1], 
            label="Predicted Non-Spam", marker="x", color="blue")
plt.plot(x_vals_train, y_vals_train, color="blue", linestyle="--", label="Decision Boundary")
plt.xlabel("Email Length (words)")
plt.ylabel("Number of Special Characters")
plt.title("Spam vs Non-Spam Classification (Test Data)")
plt.legend()
plt.show()

from sklearn.model_selection import KFold
# Cross-validation function
def cross_validate(data_spam, data_nonspam, k_values, d, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    avg_errors = []

    for k in k_values:
        errors = []

        for train_index, val_index in kf.split(data_spam):
            X_train_fold = np.vstack((data_spam[train_index], data_nonspam[train_index]))
            y_train_fold = np.hstack((np.ones(len(train_index)), np.zeros(len(train_index))))
            X_val_fold = np.vstack((data_spam[val_index], data_nonspam[val_index]))
            y_val_fold = np.hstack((np.ones(len(val_index)), np.zeros(len(val_index))))

            best_theta_fold, best_theta0_fold, _ = random_linear_classifier(
                X_train_fold[y_train_fold == 1], X_train_fold[y_train_fold == 0], k, d
            )
            errors.append(compute_error(X_val_fold[y_val_fold == 1], X_val_fold[y_val_fold == 0],
                                         best_theta_fold, best_theta0_fold))

        avg_errors.append(np.mean(errors))

    best_k = k_values[np.argmin(avg_errors)]
    return best_k

# Cross-validate to find best k
k_values = [100,1000, 10000, 20000]
best_k = cross_validate(train_spam, train_nonspam, k_values, d=2)
print(f"Best value of k: {best_k}")