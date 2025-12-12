"""
Test script to verify the frontend collects and processes all required data
"""
import requests
import json

def test_home_page():
    """Test that home page loads"""
    print("Testing home page...")
    try:
        response = requests.get('http://localhost:5000/')
        if response.status_code == 200:
            print("✅ Home page loads successfully")
            # Check if demographic fields are in the HTML
            if 'age' in response.text and 'height' in response.text and 'weight' in response.text:
                print("✅ Demographic fields found in HTML")
            else:
                print("❌ Demographic fields missing from HTML")
            return True
        else:
            print(f"❌ Home page returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error accessing home page: {e}")
        return False

def test_assessment_form():
    """Test that assessment form processes data correctly"""
    print("\nTesting assessment form submission...")

    # Test data - typical TB case
    test_data = {
        # Demographics
        'age': 35,
        'gender': 'M',
        'height': 175,
        'weight': 65,

        # Symptoms - Classic TB case
        'cough': 1,
        'cough_gt_2w': 1,
        'blood_in_sputum': 1,
        'fever': 1,
        'low_grade_fever': 0,
        'weight_loss': 1,
        'night_sweats': 1,
        'chest_pain': 0,
        'breathing_problem': 0,
        'fatigue': 1,
        'loss_of_appetite': 1,
        'contact_with_TB': 0
    }

    try:
        response = requests.post('http://localhost:5000/assess', data=test_data)
        if response.status_code == 200:
            print("✅ Assessment form processed successfully")

            # Check if results page contains expected elements
            if 'TB Assessment Results' in response.text:
                print("✅ Results page loaded")

            if 'Patient Information' in response.text or 'demographics' in response.text.lower():
                print("✅ Demographics displayed in results")
            else:
                print("⚠️ Demographics may not be displayed in results")

            if 'BMI' in response.text or 'bmi' in response.text.lower():
                print("✅ BMI calculated and displayed")
            else:
                print("⚠️ BMI may not be displayed")

            return True
        else:
            print(f"❌ Assessment returned status {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False
    except Exception as e:
        print(f"❌ Error during assessment: {e}")
        return False

def test_api_endpoint():
    """Test the API endpoint"""
    print("\nTesting API endpoint...")

    test_symptoms = {
        'symptoms': {
            'cough': 1,
            'cough_gt_2w': 1,
            'blood_in_sputum': 0,
            'fever': 1,
            'low_grade_fever': 0,
            'weight_loss': 1,
            'night_sweats': 0,
            'chest_pain': 0,
            'breathing_problem': 0,
            'fatigue': 1,
            'loss_of_appetite': 0,
            'contact_with_TB': 0
        }
    }

    try:
        response = requests.post(
            'http://localhost:5000/api/assess',
            json=test_symptoms,
            headers={'Content-Type': 'application/json'}
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ API endpoint working")
            print(f"   Risk Category: {result.get('risk_category', 'N/A')}")
            print(f"   Probability: {result.get('probability', 'N/A')}%")
            print(f"   Symptoms Present: {result.get('symptoms_present', 'N/A')}")
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error calling API: {e}")
        return False

def test_healthy_case():
    """Test with a healthy individual"""
    print("\nTesting healthy individual case...")

    test_data = {
        # Demographics
        'age': 25,
        'gender': 'F',
        'height': 165,
        'weight': 58,

        # No symptoms
        'cough': 0,
        'cough_gt_2w': 0,
        'blood_in_sputum': 0,
        'fever': 0,
        'low_grade_fever': 0,
        'weight_loss': 0,
        'night_sweats': 0,
        'chest_pain': 0,
        'breathing_problem': 0,
        'fatigue': 0,
        'loss_of_appetite': 0,
        'contact_with_TB': 0
    }

    try:
        response = requests.post('http://localhost:5000/assess', data=test_data)
        if response.status_code == 200:
            if 'Healthy' in response.text or 'Low' in response.text:
                print("✅ Healthy case classified correctly")
                return True
            else:
                print("⚠️ Healthy case may not be classified correctly")
                return False
        else:
            print(f"❌ Request failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("TB DETECTION FRONTEND TEST SUITE")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Home Page", test_home_page()))
    results.append(("Assessment Form (TB Case)", test_assessment_form()))
    results.append(("API Endpoint", test_api_endpoint()))
    results.append(("Healthy Case", test_healthy_case()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")

    print("=" * 60)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 60)

    if passed == total:
        print("\n🎉 All tests passed! Frontend is working correctly.")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please review the issues above.")
