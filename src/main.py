import argparse
import os
import sys

# Add parent directory to sys.path so `import config` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from index_manager import (
    add_document,
    get_or_create_index,
    list_documents_in_index,
    rebuild_index,
    remove_document_by_filename,
)
from llm_setup import setup_settings
from query_processor import query_anything

# Stock test queries
TESTS = [
    ("Руководитель", "Кто является научным руководителем договора №16.09-92/2024у?"),
    ("Сумма", "На какую сумму заключен договор №16.09-92/2024у?"),
    (
        "Организация",
        "Какая организация является заказчиком по договору №16.09-92/2024у?",
    ),
    ("Дата", "Когда был заключен договор №16.09-92/2024у?"),
    ("Новый договор 1", "Кто научный руководитель по договору №16.09-133/2023 у?"),
    ("Новый договор 2", "С какой организацией заключен договор №16.09-429/2024 у?"),
    ("Новый договор 3", "Какая дата у договора №16.09-261/2023 у?"),
    ("Новый договор 4", "Кто заказчик по договору №16.09-93/2023 у?"),
    ("Сотрудник 1", "В каких договорах участвует Белкин Денис Сергеевич?"),
    ("Сотрудник 2", "Какие должности занимает Жуйкова Е.В. в рабочих группах?"),
    ("Сотрудник 3", "Сколько договоров с участием Першиной А.А. есть в базе?"),
    ("Сотрудник 4", "Кто является директором ИШНКБ согласно приказам?"),
    ("Организации 1", "Какие договоры есть с ООО «РУСАВИАПРОМ»?"),
    ("Организации 2", "С какими организациями заключены договоры в 2024 году?"),
    ("Организации 3", "Есть ли договоры с АО «НПФ «Микран»?"),
    ("Даты 1", "Какие договоры заключены в марте 2024 года?"),
    ("Даты 2", "Какие приказы о назначении рабочих групп изданы в 2023 году?"),
    ("Даты 3", "Какой самый поздний договор есть в базе?"),
    ("Подразделения 1", "Какое подразделение ТПУ чаще всего участвует в договорах?"),
    ("Подразделения 2", "Кто входит в РЦАКД согласно приказам?"),
    ("Сравнение", "Сравни договоры №16.09-92/2024у и №16.09-133/2023 у"),
    ("Агрегация", "Сколько всего договоров есть в базе данных?"),
    ("Сумма общая", "Какова общая сумма всех договоров в базе?"),
    ("Сложный 1", "Кто из сотрудников участвует во всех договорах 2024 года?"),
    ("Сложный 2", "Какие договоры заключены с организациями из Новосибирска?"),
    ("Сложный 3", "Найди все документы, где упоминается Баранов Павел Федорович"),
]


def print_separator(char="=", length=70):
    print(char * length)


def run_single_test(index, test_name, test_query):
    """Run a single test query and print the result."""
    print_separator()
    print(f"ТЕСТ: {test_name}")
    print(f"Вопрос: {test_query}")
    print_separator()

    answer = query_anything(index, test_query, show_debug=True)
    print(f"\nОТВЕТ:\n{answer}\n")


def run_all_tests(index):
    """Run all stock tests sequentially."""
    print_separator()
    print("ЗАПУСК ВСЕХ СТОКОВЫХ ТЕСТОВ")
    print_separator()

    for i, (test_name, test_query) in enumerate(TESTS, 1):
        print(f"\n[{i}/{len(TESTS)}]")
        run_single_test(index, test_name, test_query)


def list_tests():
    """Print the list of available stock tests."""
    print_separator()
    print("СПИСОК ДОСТУПНЫХ ТЕСТОВ:")
    print_separator()
    for i, (test_name, test_query) in enumerate(TESTS, 1):
        print(f"{i:2d}. [{test_name}] {test_query}")
    print_separator()


def run_selected_tests(index):
    """Let user pick specific tests to run."""
    list_tests()
    print(
        "Введите номера тестов через запятую (например: 1,3,5) или 'all' для запуска всех:"
    )
    choice = input("> ").strip()

    if choice.lower() == "all":
        run_all_tests(index)
        return

    try:
        indices = [int(x.strip()) for x in choice.split(",")]
    except ValueError:
        print("Ошибка: введите корректные номера через запятую.")
        return

    for idx in indices:
        if 1 <= idx <= len(TESTS):
            test_name, test_query = TESTS[idx - 1]
            run_single_test(index, test_name, test_query)
        else:
            print(f"Тест с номером {idx} не существует, пропуск.")


def run_custom_query(index):
    """Run a single custom query from user input."""
    print_separator()
    print("ВВЕДИТЕ СВОЙ ВОПРОС:")
    print_separator()
    query = input("> ").strip()

    if not query:
        print("Пустой вопрос, отмена.")
        return

    answer = query_anything(index, query, show_debug=True)
    print(f"\nОТВЕТ:\n{answer}\n")


def run_custom_loop(index):
    """Interactive loop for multiple custom queries."""
    print_separator()
    print("РЕЖИМ СВОИХ ВОПРОСОВ (введите 'exit' или 'q' для выхода)")
    print_separator()

    while True:
        query = input("\nВопрос > ").strip()

        if query.lower() in ("exit", "q", "quit", "выход"):
            print("Выход из режима вопросов.")
            break

        if not query:
            continue

        answer = query_anything(index, query, show_debug=True)
        print(f"\nОТВЕТ:\n{answer}")


def list_documents_menu(index):
    """Menu for listing all documents in the index."""
    print_separator()
    print("СПИСОК ДОКУМЕНТОВ В ИНДЕКСЕ")
    print_separator()

    filenames = list_documents_in_index(index)

    if not filenames:
        print("Индекс пуст.")
        return

    print(f"Всего документов в индексе: {len(filenames)}\n")

    for i, filename in enumerate(filenames, 1):
        print(f"{i:4d}. {filename}")

    print_separator()


def show_menu():
    """Display the main menu."""
    print_separator()
    print("ГЛАВНОЕ МЕНЮ")
    print_separator()
    print("1. Запустить все стоковые тесты")
    print("2. Выбрать конкретные стоковые тесты")
    print("3. Задать свой вопрос (один)")
    print("4. Задать свои вопросы (цикл)")
    print("5. Показать список стоковых тестов")
    print("6. Добавить документ в индекс")
    print("7. Удалить документ из индекса")
    print("8. Показать список документов в индексе")
    print("9. Пересоздать индекс целиком (только если нужно)")
    print("0. Выход")
    print_separator()


def main():
    parser = argparse.ArgumentParser(description="Система поиска по документам")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache usage",
    )
    parser.add_argument(
        "--mode",
        type=int,
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        help="Run specific mode without menu (1-9, 0=exit)",
    )
    args = parser.parse_args()

    # Apply cache flag
    if args.no_cache:
        config.USE_CACHE = False
        print("[Кэш отключен]")
    else:
        print(f"[Кэш {'включен' if config.USE_CACHE else 'отключен'}]")

    # Initialize LLM and index
    setup_settings()
    index = get_or_create_index()

    # If mode specified via CLI, run it directly
    if args.mode is not None:
        if args.mode == 1:
            run_all_tests(index)
        elif args.mode == 2:
            run_selected_tests(index)
        elif args.mode == 3:
            run_custom_query(index)
        elif args.mode == 4:
            run_custom_loop(index)
        elif args.mode == 5:
            list_tests()
        elif args.mode == 6:
            add_document_menu(index)
        elif args.mode == 7:
            remove_document_menu(index)
        elif args.mode == 8:
            list_documents_menu(index)
        elif args.mode == 9:
            new_index = rebuild_index_menu()
            if new_index:
                index = new_index
        return

    # Interactive menu loop
    while True:
        show_menu()
        choice = input("Выберите режим > ").strip()

        if choice == "1":
            run_all_tests(index)
        elif choice == "2":
            run_selected_tests(index)
        elif choice == "3":
            run_custom_query(index)
        elif choice == "4":
            run_custom_loop(index)
        elif choice == "5":
            list_tests()
        elif choice == "6":
            add_document_menu(index)
        elif choice == "7":
            remove_document_menu(index)
        elif choice == "8":
            list_documents_menu(index)
        elif choice == "9":
            new_index = rebuild_index_menu()
            if new_index:
                index = new_index
        elif choice == "0":
            print("Выход.")
            break
        else:
            print("Неизвестный выбор, попробуйте снова.")


if __name__ == "__main__":
    main()
