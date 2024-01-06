from utils.file import ls, read_file_lines_from, write_file_lines_to, without_ext


def start():
    target_dir = "../dataset/geomagnetic"
    origin_file_names = ls(target_dir)

    for origin_file_name in origin_file_names:
        file_lines = tuple(read_file_lines_from(f"{target_dir}/{origin_file_name}"))
        cleaned_items: [[str]] = []

        for file_line in file_lines:
            raw_data_items = file_line.strip().split()
            cleaned_items.append(raw_data_items)

        current_year = int(cleaned_items[0][0])
        current_year_lines: [str] = []
        for items in cleaned_items:
            year = int(items[0])

            if year != current_year:
                write_file_lines_to(
                    f"{target_dir}/cleaned_{current_year}_"
                    f"{without_ext(origin_file_name)}.csv",
                    lines=current_year_lines
                )
                current_year_lines.clear()
                current_year = year

            current_year_lines.append(','.join(items))

        write_file_lines_to(
            f"{target_dir}/cleaned_{current_year}_"
            f"{without_ext(origin_file_name)}.csv",
            lines=current_year_lines
        )


if __name__ == '__main__':
    start()
